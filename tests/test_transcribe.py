import os
import platform
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from transcribe import (
    expand_input_patterns,
    extract_wav,
    ffmpeg_install_hint,
    load_wav_for_pyannote,
    pick_backend,
    run_diarization,
)


class PlatformSelectionTest(unittest.TestCase):
    def test_uses_mlx_on_apple_silicon(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sys, "platform", "darwin"),
            mock.patch.object(platform, "machine", return_value="arm64"),
        ):
            self.assertEqual(pick_backend(), "mlx")

    def test_uses_faster_whisper_on_windows(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(platform, "machine", return_value="AMD64"),
        ):
            self.assertEqual(pick_backend(), "faster-whisper")

    def test_environment_override_takes_priority(self) -> None:
        with mock.patch.dict(os.environ, {"STT_BACKEND": "faster-whisper"}, clear=True):
            self.assertEqual(pick_backend(), "faster-whisper")

    def test_returns_windows_ffmpeg_install_hint(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(ffmpeg_install_hint(), "winget install ffmpeg")


class ExpandInputPatternsTest(unittest.TestCase):
    def test_expands_wildcards_in_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            second = root / "b.wav"
            first = root / "a.wav"
            second.touch()
            first.touch()

            actual = expand_input_patterns([root / "*.wav"])

            self.assertEqual(actual, [first, second])

    def test_keeps_unmatched_pattern_for_missing_file_error(self) -> None:
        pattern = Path("input") / "*.missing"

        self.assertEqual(expand_input_patterns([pattern]), [pattern])

    def test_keeps_regular_path_unchanged(self) -> None:
        path = Path("input") / "meeting.mp3"

        self.assertEqual(expand_input_patterns([path]), [path])


class LoadWavForPyannoteTest(unittest.TestCase):
    def test_loads_pcm_as_normalized_channel_first_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audio.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(struct.pack("<3h", -32768, 0, 32767))

            audio = load_wav_for_pyannote(path)

            self.assertEqual(audio["sample_rate"], 16_000)
            self.assertEqual(tuple(audio["waveform"].shape), (1, 3))
            self.assertEqual(audio["waveform"][0, 0].item(), -1.0)
            self.assertEqual(audio["waveform"][0, 1].item(), 0.0)
            self.assertAlmostEqual(audio["waveform"][0, 2].item(), 1.0, places=4)


class ExtractWavTest(unittest.TestCase):
    def test_requests_portable_pcm_and_utf8_subprocess_output(self) -> None:
        src = Path("入力.mov")
        dst = Path("音声.wav")

        with mock.patch("transcribe.subprocess.run") as run:
            extract_wav(src, dst)

        run.assert_called_once_with(
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
            encoding="utf-8",
            errors="replace",
        )


class RunDiarizationTest(unittest.TestCase):
    def test_passes_decoded_audio_instead_of_a_path(self) -> None:
        audio = {"waveform": mock.sentinel.waveform, "sample_rate": 16_000}
        annotation = mock.Mock()
        annotation.itertracks.return_value = []
        result = mock.Mock(speaker_diarization=annotation)
        pipeline = mock.Mock(return_value=result)

        with (
            mock.patch("transcribe.extract_wav") as extract,
            mock.patch("transcribe.load_wav_for_pyannote", return_value=audio) as load,
        ):
            turns = run_diarization(pipeline, Path("入力.mov"), 2)

        self.assertEqual(turns, [])
        extract.assert_called_once()
        load.assert_called_once_with(extract.call_args.args[1])
        pipeline.assert_called_once_with(audio, num_speakers=2)


if __name__ == "__main__":
    unittest.main()
