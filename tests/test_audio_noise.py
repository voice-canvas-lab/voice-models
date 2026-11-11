import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audio_noise import measured_snr, mix_noise, record_seed
from check_dataset import audit


class NoiseTests(unittest.TestCase):
    def test_clean_and_multivoice_configs(self):
        from synthesize import load_config
        configs = Path(__file__).resolve().parents[1] / "configs"
        clean = load_config(configs / "pilot-clean.json")
        self.assertEqual(clean["VOICES"], ["xenia"])
        self.assertEqual(clean["NOISE_SNRS_DB"], [])
        multi = load_config(configs / "pilot-multivoice.json")
        self.assertEqual(len(multi["VOICES"]) * (1 + len(multi["NOISE_SNRS_DB"])) * 4, 36)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for levels in ([20, 20], [20, 20.000000001]):
                path.write_text(json.dumps({"VOICES": ["xenia"], "SEED": 42, "NOISE_SNRS_DB": levels}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_snr_pcm_and_peak_protection(self):
        clean = 0.99 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
        for snr in (20, 10):
            mixed, settings = mix_noise(clean, snr, 42)
            self.assertLess(settings["mixture_gain"], 1)
            self.assertLessEqual(np.max(np.abs(mixed)), 0.98000001)
            self.assertAlmostEqual(measured_snr(clean, mixed, settings["mixture_gain"]), snr, places=10)
            stream = io.BytesIO()
            sf.write(stream, mixed, 16000, format="WAV", subtype="PCM_16")
            stream.seek(0)
            pcm, _ = sf.read(stream)
            self.assertLess(abs(measured_snr(clean, pcm, settings["mixture_gain"]) - snr), 0.05)

    def test_seed_reproducibility(self):
        clean = np.ones(16000) * 0.1
        seed = record_seed(42, "sample", "xenia", 20)
        first, _ = mix_noise(clean, 20, seed)
        second, _ = mix_noise(clean, 20, seed)
        np.testing.assert_array_equal(first, second)
        self.assertNotEqual(seed, record_seed(42, "sample", "baya", 20))
        self.assertNotEqual(seed, record_seed(42, "sample", "xenia", 10))

    def test_invalid_signal(self):
        for signal in (np.zeros(100), np.array([]), np.array([np.nan]), np.ones((2, 10))):
            with self.assertRaises(ValueError):
                mix_noise(signal, 20, 42)

    def test_audit_detects_invalid_source_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            (root / "config.json").write_text(json.dumps({"VOICES": ["xenia"], "NOISE_SNRS_DB": [20], "SEED": 42}))
            (root / "samples.jsonl").write_text(json.dumps({"sample_id": "s", "group_id": "g", "split": "validation"}) + "\n")
            signal = 0.2 * np.sin(2 * np.pi * np.arange(16000) * 440 / 16000)
            sf.write(root / "audio/clean.wav", signal, 16000, subtype="PCM_16")
            clean, _ = sf.read(root / "audio/clean.wav")
            seed = record_seed(42, "s", "xenia", 20)
            mixed, aug = mix_noise(clean, 20, seed)
            sf.write(root / "audio/noisy.wav", mixed, 16000, subtype="PCM_16")
            noisy, _ = sf.read(root / "audio/noisy.wav")
            aug.update(source_audio_id="clean", measured_snr_db=measured_snr(clean, noisy, aug["mixture_gain"]))
            common = {"sample_id": "s", "voice": "xenia", "split": "validation", "seed": 42, "duration_sec": 1}
            records = [{**common, "audio_id": "clean", "path": "audio/clean.wav", "augmentations": []},
                       {**common, "audio_id": "noisy", "path": "audio/noisy.wav", "source_audio_id": "clean", "augmentations": [aug]}]
            def save():
                (root / "audio_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            save()
            self.assertTrue(audit(root)["passed"])
            aug["seed"] += 1
            save()
            with self.assertRaisesRegex(ValueError, "seed"):
                audit(root)
            aug["seed"] -= 1
            records[1]["source_audio_id"] = "noisy"
            save()
            with self.assertRaises(ValueError):
                audit(root)


if __name__ == "__main__":
    unittest.main()
