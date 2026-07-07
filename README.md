# speech-to-text

動画・音声ファイルをローカルで文字起こしする CLI。

- **エンジン**: Whisper large-v3。実行環境に応じて自動切替
  - Apple Silicon Mac: [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)(Metal GPU)
  - Windows / Linux: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)(NVIDIA GPU があれば CUDA、なければ CPU)
- **話者分離**: [pyannote.audio](https://github.com/pyannote/pyannote-audio)(`--diarize` 指定時のみ)
- **対応入力**: ffmpeg が読める形式すべて(mp4 / mov / mkv / mp3 / wav / m4a など)
- すべてローカル処理。音声データが外部に送信されることはない(モデルの初回ダウンロードのみネットワークを使用)

## 必要なもの

- [uv](https://docs.astral.sh/uv/)(Python 本体と依存は uv が自動管理)
- ffmpeg(macOS: `brew install ffmpeg` / Windows: `winget install ffmpeg` / Linux: `apt install ffmpeg` など)
- GPU で動かす場合: Apple Silicon Mac、または NVIDIA GPU(下記「Windows / Linux での GPU 利用」参照)

## セットアップ

```sh
git clone https://github.com/yone-al/speech-to-text.git
cd speech-to-text
```

これだけで完了。初回の `uv run` 時に、uv が Python 3.12 と依存ライブラリを自動でセットアップする。

## 使い方

入力ファイルはリポジトリ内に置く必要はなく、任意のパスを指定できる(例: `uv run transcribe.py ~/Movies/meeting.mp4`)。

```sh
# 基本: テキスト (.txt) を入力ファイルと同じ場所に出力
uv run transcribe.py input.mp4

# 字幕ファイル (SRT / VTT) も出力
uv run transcribe.py input.mp4 --srt --vtt

# 言語を固定(自動検出より安定する。日本語なら ja)
uv run transcribe.py input.mp4 --language ja

# 複数ファイル + 出力先指定
uv run transcribe.py a.mp3 b.mp4 --output-dir out/

# 速度優先モデルに切替(約8倍速・精度は僅かに低下)
uv run transcribe.py input.mp4 --model mlx-community/whisper-large-v3-turbo  # Mac
uv run transcribe.py input.mp4 --model large-v3-turbo                        # Windows/Linux
```

初回実行時に Whisper large-v3 モデル(約3GB)が `~/.cache/huggingface/` にダウンロードされる。2回目以降はオフラインで動作する。

処理時間の目安(M4 / 32GB): 既定の large-v3 で**実時間の 1/3〜1/5 程度**(10分の音声なら2〜3分)。turbo モデルならさらに数倍速い。NVIDIA GPU(CUDA)でも同等以上の速度が出る。CPU 実行時は実時間の数倍かかるため、`--model large-v3-turbo` などの軽いモデルを推奨。処理中は進捗バーが表示される(全 OS 共通)。

## Windows / Linux での GPU 利用

エンジン(faster-whisper)は NVIDIA GPU を自動検出して使う。

- 必要なのは **NVIDIA ドライバ + CUDA 12 対応 GPU** のみ
- CUDA 版 PyTorch は Windows では自動で入る(`pyproject.toml` の `[tool.uv.sources]` で PyTorch 公式インデックスを指定済み)。話者分離(`--diarize`)もそのまま GPU で動く
- faster-whisper の GPU 実行には cuBLAS / cuDNN 9 も必要。うまく動かない場合は [faster-whisper の README](https://github.com/SYSTRAN/faster-whisper#gpu) を参照
- **進捗バーが出た直後にエラー表示なく突然終了する場合は cuDNN 未導入が原因**(C++ 層で落ちるため Python のエラーメッセージが出せない)。上記の cuDNN セットアップを確認

GPU が見つからない場合は自動的に CPU にフォールバックする(その旨が表示される)。環境変数 `STT_BACKEND=mlx|faster-whisper` でエンジンを強制切替することもできる(mlx は Apple Silicon Mac 専用)。

## 話者分離(誰が話したか)

pyannote のモデルは利用条件同意制のため、初回のみセットアップが必要:

1. [Hugging Face](https://hf.co/settings/tokens) で read 権限のトークンを作成
2. [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) で利用条件に同意
   - **ここを飛ばすとモデル取得が 403 エラーで失敗する**(エラーメッセージに手順を再表示する)
3. トークンを登録: `uv run hf auth login`(ホームディレクトリに保存され、リポジトリ外なので安全)
   - CI などでは環境変数 `HF_TOKEN` または `--hf-token` 引数でも可(引数 → 環境変数 → login の順で優先)

`--diarize` の初回実行時は pyannote のモデル(数百MB)が追加でダウンロードされる。

```sh
uv run transcribe.py meeting.mp4 --diarize

# 話者数が分かっている場合は指定すると精度が上がる
uv run transcribe.py meeting.mp4 --diarize --num-speakers 3
```

出力例:

```
[SPEAKER_00] お疲れさまです。始めましょうか。
[SPEAKER_01] はい、お願いします。
```

## 精度のためのヒント

- 言語が分かっているなら `--language ja` を付ける(自動検出の誤りを防ぐ)
- 会議録音など話者数が既知なら `--num-speakers N` を付ける
- 極端に音量が小さい・ノイズが多い音源は、事前に ffmpeg でノーマライズすると改善することがある:
  `ffmpeg -i in.mp4 -af loudnorm out.wav`

## 既知の制限

- **無音区間で幻覚が出ることがある**: Whisper は長い無音から「ご視聴ありがとうございました」のような実在しない文を生成することがある(学習データ由来の既知問題)。動画末尾の無音などで出やすい
- **話者分離は合成音声に弱い**: 話者識別モデルは人間の声で学習されているため、TTS 音声では話者を正しく区別できないことがある。実録音では問題ない
- **Intel Mac は CPU 実行になる**: Apple Silicon(MLX)でも NVIDIA GPU(CUDA)でもないため。動作はするが遅い
- **Windows ARM64(Snapdragon 等)は非対応**: エンジン(CTranslate2)に ARM64 Windows 用のビルドがないため
