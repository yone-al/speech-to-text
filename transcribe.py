#!/usr/bin/env -S uv run
"""動画・音声ファイルをローカルで文字起こしする CLI。

Whisper large-v3 (mlx-whisper, Apple Silicon GPU) で認識し、
オプションで pyannote.audio による話者分離を行う。

使い方:
    uv run transcribe.py input.mp4                  # テキスト (.txt) を出力
    uv run transcribe.py input.mp4 --srt --vtt      # 字幕ファイルも出力
    uv run transcribe.py input.mp4 --language ja    # 言語を固定(既定: 自動検出)
    uv run transcribe.py input.mp4 --diarize        # 話者分離付き(要 HF トークン)
    uv run transcribe.py *.mp3 --output-dir out/    # 複数ファイル・出力先指定
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 精度優先で large-v3(非 turbo)を既定にしている。速度優先なら --model で
# mlx-community/whisper-large-v3-turbo に切替可能(約8倍速・精度は僅かに低下)。
DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"

# pyannote.audio 4.x 用の現行パイプライン。gated モデルのため HF トークンと
# モデルページ (https://hf.co/pyannote/speaker-diarization-community-1) での同意が必要。
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# 話者分離のセットアップ手順(トークン未設定・モデル未同意のエラーで表示する)
DIARIZE_SETUP_GUIDE = (
    "  1. https://hf.co/settings/tokens でトークンを作成(read 権限)\n"
    f"  2. https://hf.co/{DIARIZATION_MODEL} で利用条件に同意\n"
    "  3. `uv run hf auth login` でトークンを登録(または HF_TOKEN 環境変数 / --hf-token)"
)

# (開始秒, 終了秒, 話者ラベル) — pyannote の出力
Turn = tuple[float, float, str]


# ---------------------------------------------------------------- 文字起こし


def transcribe_file(path: Path, model: str, language: str | None) -> dict:
    """mlx-whisper で認識し、Whisper 形式の結果 dict を返す。

    入力は ffmpeg がデコードできる形式ならなんでもよい(動画も可)。
    mlx_whisper が内部で ffmpeg を呼んで音声トラックを取り出すため。
    """
    # import が遅い(数秒)ので、実際に使う関数内で読み込む
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        language=language,  # None なら自動検出
        verbose=False,  # False: 進捗バーのみ表示
    )


# ---------------------------------------------------------------- 話者分離


def resolve_hf_token(cli_token: str | None) -> str | None:
    """HF トークンを CLI 引数 → HF_TOKEN 環境変数 → `hf auth login` の保存分の順で探す。"""
    if cli_token:
        return cli_token
    # get_token() は HF_TOKEN 環境変数と login 済みトークンの両方を見てくれる
    from huggingface_hub import get_token

    return get_token()


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
                str(dst),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit(
            "エラー: ffmpeg が見つかりません。`brew install ffmpeg` でインストールしてください"
        )
    except subprocess.CalledProcessError as e:
        # 失敗理由(音声トラックなし・ファイル破損など)は stderr にしか出ない
        raise RuntimeError(f"音声抽出に失敗しました:\n{e.stderr.strip()}") from e


def load_diarization_pipeline(token: str):
    """pyannote の話者分離パイプラインを読み込む(1回読み込んで全ファイルで使い回す)。

    gated モデルのため、トークン不備や利用条件未同意はここで失敗する。
    文字起こし(長時間)の後に気づくと待ち時間が無駄になるので、呼び出し側は
    処理開始前にこれを呼ぶこと。
    """
    import torch
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except Exception as e:
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

    if torch.backends.mps.is_available():
        # Apple Silicon の GPU (Metal) で実行
        pipeline.to(torch.device("mps"))
    else:
        print("  (GPU 非対応環境のため CPU で実行します — 時間がかかる場合があります)")
    return pipeline


def run_diarization(pipeline, path: Path, num_speakers: int | None) -> list[Turn]:
    """話者分離を実行し、話者区間のリストを返す。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav = Path(tmpdir) / "audio.wav"
        extract_wav(path, wav)
        kwargs = {"num_speakers": num_speakers} if num_speakers is not None else {}
        # pyannote 4.x は DiarizeOutput を返す(3.x は Annotation を直接返していた)
        annotation = pipeline(str(wav), **kwargs).speaker_diarization

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
    out_dir = Path(args.output_dir) if args.output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / path.stem

    print(f"* 文字起こし中: {path.name} (model: {args.model})")
    t0 = time.time()
    result = transcribe_file(path, args.model, args.language)
    segments = result["segments"]
    print(f"  完了 ({time.time() - t0:.1f}s, 言語: {result['language']})")

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
    parser = argparse.ArgumentParser(
        description="動画・音声ファイルをローカルで文字起こしする",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="動画または音声ファイル")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper モデル(速度優先: mlx-community/whisper-large-v3-turbo)",
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

    if not args.diarize and (args.num_speakers is not None or args.hf_token):
        print("警告: --num-speakers / --hf-token は --diarize 指定時のみ使われます")

    # 入力の一括検証: 処理を始めてから typo に気づくと、それまでの処理時間が無駄になる
    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        sys.exit("エラー: ファイルが見つかりません: " + ", ".join(missing))

    # 出力先の衝突検知: a.mp3 と a.mp4 を同時に渡すと、どちらの結果も
    # a.txt に書かれて片方が消えるのを防ぐ
    planned: dict[Path, Path] = {}
    for path in args.inputs:
        out_dir = Path(args.output_dir) if args.output_dir else path.parent
        base = out_dir / path.stem
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
        except Exception as e:
            failed.append(path.name)
            print(f"エラー: {path.name} の処理に失敗しました: {e}", file=sys.stderr)
    if failed:
        sys.exit(
            f"{len(failed)}/{len(args.inputs)} 件が失敗しました: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
