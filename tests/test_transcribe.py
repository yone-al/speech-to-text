import struct
import tempfile
import unittest
import wave
from pathlib import Path

from transcribe import expand_input_patterns, load_wav_for_pyannote


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


if __name__ == "__main__":
    unittest.main()
