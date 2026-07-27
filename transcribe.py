#!/usr/bin/env -S uv run
"""動画・音声ファイルをローカルで文字起こしする CLI。

Whisper large-v3 で認識し、オプションで pyannote.audio による話者分離を行う。
エンジンは実行環境で自動切替:
    - Apple Silicon Mac: mlx-whisper (Metal GPU)
    - Windows / Linux:   faster-whisper (NVIDIA GPU があれば CUDA、なければ CPU)

使い方:
    uv run transcribe.py                            # input/ 内をまとめて処理し output/ に出力
    uv run transcribe.py input.mp4                  # テキスト (.txt) を出力
    uv run transcribe.py input.mp4 --srt --vtt      # 字幕ファイルも出力
    uv run transcribe.py input.mp4 --language ja    # 言語を固定(既定: 自動検出)
    uv run transcribe.py input.mp4 --diarize        # 話者分離付き(要 HF トークン)
    uv run transcribe.py *.mp3 --output-dir out/    # 複数ファイル・出力先指定
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import wave
from pathlib import Path

# 精度優先で large-v3(非 turbo)を既定にしている。速度優先なら --model で
# turbo 系(mlx: mlx-community/whisper-large-v3-turbo / faster-whisper:
# large-v3-turbo)に切替可能(約8倍速・精度は僅かに低下)。
DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-mlx",
    "faster-whisper": "large-v3",
}

# pyannote.audio 4.x 用の現行パイプライン。gated モデルのため HF トークンと
# モデルページ (https://hf.co/pyannote/speaker-diarization-community-1) での同意が必要。
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# 話者分離のセットアップ手順(トークン未設定・モデル未同意のエラーで表示する)
DIARIZE_SETUP_GUIDE = (
    "  1. https://hf.co/settings/tokens でトークンを作成(read 権限)\n"
    f"  2. https://hf.co/{DIARIZATION_MODEL} で利用条件に同意\n"
    "  3. `uv run hf auth login` でトークンを登録(または HF_TOKEN 環境変数 / --hf-token)"
)

# 繰り返し・幻覚の抑制オプション(mlx-whisper / faster-whisper 共通のシグネチャ)。
# - condition_on_previous_text=False: 直前窓のテキストを次窓のプロンプトに使わない。
#   既定の True は一度繰り返しが始まると後続の窓に伝播して自己増幅するため、
#   同じ文が延々と続く出力の主要因になる(用語の一貫性より安定性を優先)。
# - hallucination_silence_threshold: 発話終端の後にこの秒数以上の無音が残る窓を
#   幻覚の温床とみなしてスキップする。word_timestamps=True が実装上の前提
#   (単語境界がないと無音区間を判定できない)。
ANTI_HALLUCINATION_OPTIONS = {
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "hallucination_silence_threshold": 2.0,
}

# (開始秒, 終了秒, 話者ラベル) — pyannote の出力
Turn = tuple[float, float, str]

# 入力を省略したときに使う既定ディレクトリ。input/ に置いた音声・動画を
# まとめて文字起こしし、結果を output/ に書き出す(単体ファイル指定時は従来通り)。
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"

# ディレクトリを入力に指定したときに拾う拡張子(ffmpeg が読める代表的な音声・動画)。
MEDIA_EXTENSIONS = {
    # 動画
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".ts",
    ".mpg",
    ".mpeg",
    ".wmv",
    # 音声
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
}


def collect_media_in_dir(directory: Path) -> list[Path]:
    """ディレクトリ直下(サブフォルダは見ない)の音声・動画ファイルを名前順で集める。

    隠しファイルは除外する。特に SMB/FAT 上で macOS が作る AppleDouble
    (._movie.mp4 など)は拡張子が本体と同じで、ffmpeg が読めず失敗するため。
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in MEDIA_EXTENSIONS
    )


def expand_input_patterns(paths: list[Path]) -> list[Path]:
    """入力パスに含まれるワイルドカードを OS に依存せず展開する。

    macOS/Linux の一般的なシェルは ``*.mp3`` をコマンド実行前に展開する一方、
    Windows PowerShell はネイティブコマンドへの引数をそのまま渡す。その差を
    CLI 側で吸収する。マッチしないパターンは残し、後段の一括検証で通常の
    「ファイルが見つかりません」エラーとして報告する。
    """
    expanded: list[Path] = []
    for path in paths:
        pattern = os.fspath(path)
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else []
        if matches:
            expanded.extend(Path(match) for match in matches)
        else:
            expanded.append(path)
    return expanded


def output_base(path: Path, output_dir: str | None) -> Path:
    """出力ファイルのパス(拡張子抜き)を決める。

    出力先の指定がなければ入力ファイルと同じ場所。出力衝突検知と実際の
    書き込みの両方がこの関数を使うことで、検査対象と書き込み先のずれを防ぐ。
    """
    out_dir = Path(output_dir) if output_dir else path.parent
    return out_dir / path.stem


# ---------------------------------------------------------------- 文字起こし


def pick_backend() -> str:
    """実行環境に合った文字起こしエンジンを選ぶ。

    Apple Silicon は MLX (Metal GPU)、それ以外は faster-whisper。
    STT_BACKEND 環境変数(mlx / faster-whisper)で強制切替も可能。
    """
    forced = os.environ.get("STT_BACKEND")
    if forced:
        return forced
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster-whisper"


def transcribe_file(path: Path, model: str, language: str | None, backend: str) -> dict:
    """指定エンジンで認識し、Whisper 形式の結果 dict を返す。

    入力は ffmpeg がデコードできる形式ならなんでもよい(動画も可)。
    どちらのエンジンも内部で ffmpeg を呼んで音声トラックを取り出すため。
    """
    if backend == "mlx":
        return transcribe_mlx(path, model, language)
    return transcribe_faster_whisper(path, model, language)


def transcribe_mlx(path: Path, model: str, language: str | None) -> dict:
    """mlx-whisper (Apple Silicon の Metal GPU) で認識する。"""
    # import が遅い(数秒)ので、実際に使う関数内で読み込む
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        language=language,  # None なら自動検出
        verbose=False,  # False: 進捗バーのみ表示
        **ANTI_HALLUCINATION_OPTIONS,
    )


# 読み込み済みの faster-whisper モデル(モデル名 → WhisperModel)
_fw_models: dict[str, object] = {}


def transcribe_faster_whisper(path: Path, model: str, language: str | None) -> dict:
    """faster-whisper (CTranslate2) で認識し、mlx-whisper と同じ形の dict に揃える。

    NVIDIA GPU (CUDA) があれば float16 で GPU 実行、なければ int8 量子化で CPU 実行。
    """
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except ImportError:
        # pyproject のマーカーにより Apple Silicon にはインストールされない
        sys.exit(
            "エラー: faster-whisper がインストールされていません。\n"
            "Apple Silicon Mac では mlx エンジン(既定)を使用してください"
        )
    from tqdm import tqdm

    whisper = _fw_models.get(model)
    if whisper is None:
        if ctranslate2.get_cuda_device_count() > 0:
            # 注意: cuDNN 未導入でもここは成功し、認識実行時に C++ 層でプロセスが
            # 即死する(Python から捕捉不能)。突然終了する場合は README の GPU 節を参照
            whisper = WhisperModel(model, device="cuda", compute_type="float16")
        else:
            print(
                "  (NVIDIA GPU を検出できないため CPU で実行します — 時間がかかります。"
                "GPU があるのに表示される場合はドライバ等を確認 — README 参照)"
            )
            whisper = WhisperModel(model, device="cpu", compute_type="int8")
        # バッチ処理でファイルごとに数 GB のモデルを再ロードしないようキャッシュ
        # (mlx 側はライブラリ内部でキャッシュされるため不要)
        _fw_models[model] = whisper

    # transcribe() は遅延評価のジェネレータを返し、回した分だけ認識が進む。
    # 「処理済みの音声秒数 / 総秒数」で進捗バーを描画しながら回収する
    # vad_filter: 無音・非音声区間を認識前に除去する(Silero VAD)。幻覚の
    # 発生源を減らせる。mlx-whisper に相当機能はないため faster-whisper のみ
    segments_iter, info = whisper.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        **ANTI_HALLUCINATION_OPTIONS,
    )
    segments = []
    with tqdm(total=round(info.duration, 2), unit="s", leave=False) as bar:
        for s in segments_iter:
            segments.append({"start": s.start, "end": s.end, "text": s.text})
            bar.update(round(s.end - bar.n, 2))
        if bar.total is not None and bar.n < bar.total:
            # 最後の認識セグメント終端が総尺より前の場合でも、処理完了時は
            # 進捗バーを 100% にしてから閉じる。
            bar.update(bar.total - bar.n)
    return {"segments": segments, "language": info.language}


# ---------------------------------------------------------------- 話者分離


def resolve_hf_token(cli_token: str | None) -> str | None:
    """HF トークンを CLI 引数 → HF_TOKEN 環境変数 → `hf auth login` の保存分の順で探す。"""
    if cli_token:
        return cli_token
    # get_token() は HF_TOKEN 環境変数と login 済みトークンの両方を見てくれる
    from huggingface_hub import get_token

    return get_token()


def ffmpeg_install_hint() -> str:
    """OS に応じた ffmpeg のインストールコマンド例を返す。"""
    hints = {"darwin": "brew install ffmpeg", "win32": "winget install ffmpeg"}
    return hints.get(sys.platform, "apt install ffmpeg など")


def extract_wav(src: Path, dst: Path) -> None:
    """ffmpeg で 16kHz モノラル WAV に変換する(pyannote の入力用)。"""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(dst),
            ],
            check=True,
            capture_output=True,
            text=True,
            # Windows の既定ロケールが CP932 でも、ffmpeg はパス等を UTF-8 で
            # 出力することがある。デコード失敗で reader thread を落とさない。
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        sys.exit(
            f"エラー: ffmpeg が見つかりません。`{ffmpeg_install_hint()}` でインストールしてください"
        )
    except subprocess.CalledProcessError as e:
        # 失敗理由(音声トラックなし・ファイル破損など)は stderr にしか出ない
        raise RuntimeError(f"音声抽出に失敗しました:\n{e.stderr.strip()}") from e


def load_wav_for_pyannote(path: Path) -> dict:
    """16-bit PCM WAV を pyannote のメモリ入力形式に読み込む。

    pyannote 4.x のパス入力は TorchCodec によるデコードを使うが、Windows では
    PyTorch・TorchCodec・ffmpeg DLL の組み合わせによって読み込みに失敗しやすい。
    直前に ffmpeg で形式を固定した WAV を標準ライブラリで読み、TorchCodec を
    通さず ``{"waveform": Tensor, "sample_rate": int}`` として渡す。
    """
    import torch

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(
            f"話者分離用 WAV のサンプル幅が不正です: {sample_width * 8} bit"
        )

    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16)
    waveform = samples.reshape(-1, channels).T.to(torch.float32) / 32768.0
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_diarization_pipeline(token: str):
    """pyannote の話者分離パイプラインを読み込む(1回読み込んで全ファイルで使い回す)。

    gated モデルのため、トークン不備や利用条件未同意はここで失敗する。
    文字起こし(長時間)の後に気づくと待ち時間が無駄になるので、呼び出し側は
    処理開始前にこれを呼ぶこと。
    """
    import torch

    # パス入力用 TorchCodec の警告は、run_diarization でメモリ入力を渡すため該当しない。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)\s*torchcodec is not installed correctly.*",
            category=UserWarning,
            module="pyannote.audio.core.io",
        )
        from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except Exception as e:  # noqa: BLE001 — 外部ライブラリの例外型が一定しない
        sys.exit(
            f"エラー: 話者分離モデルの取得に失敗しました: {e}\n"
            "モデルページで利用条件に同意していない可能性があります。\n"
            + DIARIZE_SETUP_GUIDE
        )
    if pipeline is None:
        # 一部バージョンは利用条件未同意のとき例外ではなく None を返す
        sys.exit(
            "エラー: 話者分離モデルを取得できませんでした。\n" + DIARIZE_SETUP_GUIDE
        )

    if torch.cuda.is_available():
        # NVIDIA GPU (CUDA) で実行。Windows/Linux は CUDA 版 torch が必要 (README 参照)
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        # Apple Silicon の GPU (Metal) で実行
        pipeline.to(torch.device("mps"))
    else:
        print("  (GPU 非対応環境のため CPU で実行します — 時間がかかる場合があります)")
    return pipeline


def run_diarization(pipeline, path: Path, num_speakers: int | None) -> list[Turn]:
    """話者分離を実行し、話者区間のリストを返す。"""
    # ignore_cleanup_errors: Windows ではウイルス対策等が wav を開いたままにして
    # 削除に失敗することがあるため、後始末の失敗で処理全体を落とさない
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        wav = Path(tmpdir) / "audio.wav"
        extract_wav(path, wav)
        audio = load_wav_for_pyannote(wav)
        kwargs = {"num_speakers": num_speakers} if num_speakers is not None else {}
        # pyannote 4.x は DiarizeOutput を返す(3.x は Annotation を直接返していた)
        annotation = pipeline(audio, **kwargs).speaker_diarization

    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def assign_speaker(seg_start: float, seg_end: float, turns: list[Turn]) -> str:
    """Whisper の1セグメント (seg_start〜seg_end 秒) に話者ラベルを割り当てる。

    turns は (開始秒, 終了秒, 話者ラベル) のリスト。Whisper のセグメント境界と
    pyannote の話者区間は一致しないため、話者ごとの重なり時間を合計し、
    最も長く重なった話者を採用する(最大重なり方式)。
    どの区間とも重ならない場合は "UNKNOWN" を返す。
    """
    overlap_by_speaker: dict[str, float] = {}
    for turn_start, turn_end, speaker in turns:
        overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
        if overlap > 0:
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
    if not overlap_by_speaker:
        return "UNKNOWN"
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def merge_speakers(segments: list[dict], turns: list[Turn]) -> None:
    """各セグメントに speaker キーを付与する(segments を直接更新)。"""
    for seg in segments:
        seg["speaker"] = assign_speaker(seg["start"], seg["end"], turns)


# ---------------------------------------------------------------- 出力


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """秒数を SRT/VTT のタイムスタンプ形式 (HH:MM:SS,mmm) にする。

    区切り文字だけ規格差がある: SRT はカンマ、VTT はピリオド。
    """
    ms = round(seconds * 1000)  # 全体をミリ秒に直してから桁ごとに割っていく
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def segment_text(seg: dict) -> str:
    """セグメント本文。話者ラベルがあれば先頭に付ける。"""
    text = seg["text"].strip()
    if seg.get("speaker"):
        return f"[{seg['speaker']}] {text}"
    return text


def write_txt(segments: list[dict], path: Path) -> None:
    lines = [segment_text(seg) for seg in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(segments: list[dict], path: Path) -> None:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"], sep=",")
        end = format_timestamp(seg["end"], sep=",")
        blocks.append(f"{i}\n{start} --> {end}\n{segment_text(seg)}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(segments: list[dict], path: Path) -> None:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp(seg["start"], sep=".")
        end = format_timestamp(seg["end"], sep=".")
        blocks.append(f"{start} --> {end}\n{segment_text(seg)}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------- メイン


def process(path: Path, args: argparse.Namespace, diarization_pipeline=None) -> None:
    base = output_base(path, args.output_dir)
    base.parent.mkdir(parents=True, exist_ok=True)

    print(f"* 文字起こし中: {path.name} (engine: {args.backend}, model: {args.model})")
    t0 = time.time()
    result = transcribe_file(path, args.model, args.language, args.backend)
    segments = result["segments"]
    print(f"  完了 ({time.time() - t0:.1f}s, 言語: {result['language']})")
    if not segments:
        print("  (認識されたセグメントは 0 件でした — 無音の可能性があります)")

    # 後段の話者分離が失敗しても文字起こし結果が失われないよう、まず保存する
    # (話者分離が成功したら話者ラベル付きで上書きされる)
    write_txt(segments, base.with_suffix(".txt"))

    if diarization_pipeline is not None:
        print("* 話者分離中...")
        t0 = time.time()
        turns = run_diarization(diarization_pipeline, path, args.num_speakers)
        merge_speakers(segments, turns)
        n_speakers = len({s for _, _, s in turns})
        print(f"  完了 ({time.time() - t0:.1f}s, 話者数: {n_speakers})")

    write_txt(segments, base.with_suffix(".txt"))
    print(f"  -> {base.with_suffix('.txt')}")
    if args.srt:
        write_srt(segments, base.with_suffix(".srt"))
        print(f"  -> {base.with_suffix('.srt')}")
    if args.vtt:
        write_vtt(segments, base.with_suffix(".vtt"))
        print(f"  -> {base.with_suffix('.vtt')}")


def positive_int(value: str) -> int:
    """argparse 用: 1 以上の整数のみ受け付ける。"""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("1 以上の整数を指定してください")
    return n


def main() -> None:
    if sys.platform == "win32":
        # 日本語 Windows でリダイレクト時に stdout が cp932 になり、
        # cp932 に無い文字(— など)で UnicodeEncodeError になるのを防ぐ。
        # コンソール無し起動 (pythonw) では stream が None のことがあるため防御する
        for stream in (sys.stdout, sys.stderr):
            # 失敗しても文字化け防止が効かないだけなので処理は続行してよい
            with contextlib.suppress(AttributeError, ValueError, OSError):
                stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="動画・音声ファイルをローカルで文字起こしする",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "動画・音声ファイル、ワイルドカード、またはそれらを含むディレクトリ。"
            "未指定なら input/ 内を処理して output/ に出力"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper モデル(既定: エンジンに応じた large-v3。速度優先なら turbo 系)",
    )
    parser.add_argument(
        "--language", default=None, help="言語コード (ja, en など)。未指定なら自動検出"
    )
    parser.add_argument("--srt", action="store_true", help="SRT 字幕を出力する")
    parser.add_argument("--vtt", action="store_true", help="VTT 字幕を出力する")
    parser.add_argument("--diarize", action="store_true", help="話者分離を行う")
    parser.add_argument(
        "--num-speakers",
        type=positive_int,
        default=None,
        help="話者数が分かっている場合に指定(精度向上)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face トークン(未指定なら HF_TOKEN 環境変数)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="出力先ディレクトリ(既定: 入力と同じ場所)"
    )
    args = parser.parse_args()

    # 入力未指定なら「input/ を指定された」ものとして扱い、output/ に出力する
    # (収集や空チェックは下のディレクトリ展開に一本化する)。
    if not args.inputs:
        default_input_dir = Path(DEFAULT_INPUT_DIR)
        if default_input_dir.exists() and not default_input_dir.is_dir():
            sys.exit(f"エラー: {DEFAULT_INPUT_DIR} がディレクトリではありません")
        if not default_input_dir.is_dir():
            sys.exit(
                f"エラー: 入力が指定されておらず、{DEFAULT_INPUT_DIR}/ も見つかりません。\n"
                f"{DEFAULT_INPUT_DIR}/ を作成して音声・動画ファイルを置くか、"
                "ファイルやディレクトリを直接指定してください"
            )
        args.inputs = [default_input_dir]
        if args.output_dir is None:
            args.output_dir = DEFAULT_OUTPUT_DIR

    # PowerShell は *.mp3 などを展開しないため、まず CLI 側で展開する。
    args.inputs = expand_input_patterns(args.inputs)

    # ディレクトリが指定された場合は、その直下の音声・動画ファイルに展開する。
    expanded_inputs: list[Path] = []
    empty_dirs: list[str] = []
    for path in args.inputs:
        if path.is_dir():
            media_files = collect_media_in_dir(path)
            if not media_files:
                empty_dirs.append(str(path))
            expanded_inputs.extend(media_files)
        else:
            expanded_inputs.append(path)
    # ディレクトリと中のファイルを両方指定した場合などの重複を除く(順序は維持)。
    # resolve() はシンボリックリンクの解決で出力先や表示名まで変えてしまうため、
    # 指定されたパスのまま比較する(表記違いの同一ファイルは出力衝突検知が捕まえる)
    args.inputs = list(dict.fromkeys(expanded_inputs))
    if empty_dirs:
        print(
            "警告: 対応する音声・動画ファイルがないディレクトリをスキップしました: "
            + ", ".join(empty_dirs),
            file=sys.stderr,
        )
    if not args.inputs:
        sys.exit("エラー: 処理対象の音声・動画ファイルがありません")

    # 実行環境に合ったエンジンと既定モデルを決める
    args.backend = pick_backend()
    if args.backend not in DEFAULT_MODELS:
        sys.exit(
            f"エラー: 未知のエンジンです: {args.backend}"
            f"(STT_BACKEND には {' / '.join(DEFAULT_MODELS)} を指定)"
        )
    if args.backend == "mlx" and (
        sys.platform != "darwin" or platform.machine() != "arm64"
    ):
        # mlx-whisper は Apple Silicon 以外にはインストールされない(pyproject のマーカー)
        sys.exit(
            "エラー: mlx エンジンは Apple Silicon Mac 専用です(STT_BACKEND を確認)"
        )
    if args.model is None:
        args.model = DEFAULT_MODELS[args.backend]

    if not args.diarize and (args.num_speakers is not None or args.hf_token):
        print(
            "警告: --num-speakers / --hf-token は --diarize 指定時のみ使われます",
            file=sys.stderr,
        )

    # 入力の一括検証: 処理を始めてから typo に気づくと、それまでの処理時間が無駄になる
    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        sys.exit("エラー: ファイルが見つかりません: " + ", ".join(missing))

    # ffmpeg は文字起こし(エンジン内部)と話者分離の両方で必須。
    # 長時間処理の後に「ffmpeg がない」で成果を失わないよう最初に確認する
    if shutil.which("ffmpeg") is None:
        sys.exit(
            f"エラー: ffmpeg が見つかりません。`{ffmpeg_install_hint()}` でインストールしてください"
        )

    # 出力先の衝突検知: a.mp3 と a.mp4 を同時に渡すと、どちらの結果も
    # a.txt に書かれて片方が消えるのを防ぐ
    planned: dict[Path, Path] = {}
    for path in args.inputs:
        base = output_base(path, args.output_dir)
        if base in planned:
            sys.exit(
                f"エラー: {planned[base]} と {path} は出力先が同じ ({base}.txt) になります。\n"
                "ファイル名を変えるか、別々に実行してください"
            )
        planned[base] = path

    diarization_pipeline = None
    if args.diarize:
        token = resolve_hf_token(args.hf_token)
        if not token:
            sys.exit(
                "エラー: --diarize には Hugging Face トークンが必要です。\n"
                + DIARIZE_SETUP_GUIDE
            )
        print("* 話者分離モデルを準備中...")
        diarization_pipeline = load_diarization_pipeline(token)

    # バッチ利用を想定し、1ファイルの失敗で全体を止めない
    failed: list[str] = []
    for path in args.inputs:
        try:
            process(path, args, diarization_pipeline)
        except Exception as e:  # noqa: BLE001 — 1件の失敗でバッチ全体を止めない
            failed.append(path.name)
            print(f"エラー: {path.name} の処理に失敗しました: {e}", file=sys.stderr)
    if failed:
        sys.exit(
            f"{len(failed)}/{len(args.inputs)} 件が失敗しました: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
