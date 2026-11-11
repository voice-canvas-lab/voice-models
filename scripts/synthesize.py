"""Technical Silero V5 pilot: samples JSONL -> checked WAV and manifest."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf
import torch

from input_validation import load_config, load_samples
from recognition_config import validate_tts

from audio_noise import (POWER_RULE, SNR_RULE, SNR_TOLERANCE_DB, measured_snr,
                         mix_noise, record_seed)


MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"
SOURCE_RATE = 48000
OUTPUT_RATE = 16000


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def check_wav(path):
    info = sf.info(path)
    audio, rate = sf.read(path, dtype="float64", always_2d=True)
    if (info.format, info.subtype, info.channels, rate) != ("WAV", "PCM_16", 1, OUTPUT_RATE):
        raise ValueError(f"Unexpected WAV format: {path}")
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    duration = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    if not 0.2 <= duration <= 30 or rms < 1e-5 or clipped > 0.001:
        raise ValueError(f"Audio QC failed: {path}: duration={duration}, rms={rms}, clipped={clipped}")
    return {"duration_sec": duration, "rms": rms, "peak": peak, "clipped_fraction": clipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/silero"))
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/pilot-clean.json")
    args = parser.parse_args()
    samples = load_samples(args.samples)
    config = load_config(args.config)
    tts = config.get("TTS")
    if tts is not None:
        validate_tts(tts)
    # Exclusive directory creation also prevents accidental partial-run overwrites.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.cache_dir / "v5_ru.pt"
    if not model_path.exists():
        temporary = args.cache_dir / "v5_ru.pt.download"
        torch.hub.download_url_to_file(MODEL_URL, str(temporary))
        temporary.replace(model_path)
    if tts is not None and sha256(model_path) != tts["model_sha256"]:
        raise ValueError("Cached/downloaded Silero model SHA256 differs from configured revision")
    threads = tts["cpu_threads"] if tts else 4
    torch.set_num_threads(threads)
    torch.manual_seed(config["SEED"])
    model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    settings = {"sample_rate": SOURCE_RATE,
                "put_accent": tts["put_accent"] if tts else True,
                "put_yo": tts["put_yo"] if tts else True}
    write_json(args.output / "tts_settings.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "Silero TTS V5", "model_id": "v5_ru", "model_url": MODEL_URL,
        "model_sha256": sha256(model_path), "device": "cpu", "seed": config["SEED"],
        "config": config, "config_sha256": sha256(args.config),
        "noise": {"type": "artificial_white_gaussian", "power_rule": POWER_RULE,
                  "snr_measurement_rule": SNR_RULE, "snr_tolerance_db": SNR_TOLERANCE_DB,
                  "source": "Synthetic RNG; not a recording of a real acoustic environment"},
        "threads": threads, "synthesis": settings, "output_sample_rate": OUTPUT_RATE,
        "resampling": "scipy.signal.resample_poly(up=1, down=3)",
        "peak_protection": "If peak >= 0.999, scale to 0.98; gain recorded per file in audio_checks.json",
        "output_format": "WAV mono PCM_16", "samples_sha256": sha256(args.samples),
        "script_sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
        "qc_thresholds": {"min_duration_sec": 0.2, "max_duration_sec": 30,
                          "min_rms": 1e-5, "max_clipped_fraction": 0.001,
                          "clipping_amplitude": 0.999},
    })
    write_json(args.output / "config.json", config)
    # Snapshot preserves the sample_id -> reference text relation in output root.
    with (args.output / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    qc = []
    with (args.output / "audio_manifest.jsonl").open("x", encoding="utf-8") as manifest:
        for index, sample in enumerate(samples, 1):
            for voice in config["VOICES"]:
                audio_id = f"audio_{index:04d}_{voice}_clean"
                relative_path = f"audio/{audio_id}.wav"
                tts_seed = record_seed(config["SEED"], sample["sample_id"], voice, 0)
                torch.manual_seed(tts_seed)
                with torch.inference_mode():
                    signal = model.apply_tts(text=sample["text"], speaker=voice, **settings)
                signal = signal.detach().cpu().numpy()
                if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
                    raise ValueError("Invalid synthesized signal")
                signal = resample_poly(signal, up=1, down=3)
                peak = float(np.max(np.abs(signal)))
                gain = 0.98 / peak if peak >= 0.999 else 1.0
                path = args.output / relative_path
                sf.write(path, signal * gain, OUTPUT_RATE, subtype="PCM_16", format="WAV")
                result = check_wav(path)
                qc.append({"audio_id": audio_id, "pre_pcm_peak": peak,
                           "output_gain": gain, "tts_seed": tts_seed, **result})
                common = {"sample_id": sample["sample_id"], "voice": voice,
                          "sample_rate": OUTPUT_RATE, "split": sample["split"],
                          "seed": config["SEED"], "tts_seed": tts_seed}
                manifest.write(json.dumps({**common, "audio_id": audio_id,
                    "path": relative_path, "duration_sec": result["duration_sec"],
                    "augmentations": [], "source_audio_id": None,
                }, ensure_ascii=False) + "\n")
                clean, _ = sf.read(path, dtype="float64")
                for snr in config["NOISE_SNRS_DB"]:
                    noisy_id = f"audio_{index:04d}_{voice}_snr{snr:g}"
                    noisy_path = f"audio/{noisy_id}.wav"
                    noise_seed = record_seed(config["SEED"], sample["sample_id"], voice, snr)
                    mixture, augmentation = mix_noise(clean, snr, noise_seed)
                    sf.write(args.output / noisy_path, mixture, OUTPUT_RATE, subtype="PCM_16", format="WAV")
                    noisy_pcm, _ = sf.read(args.output / noisy_path, dtype="float64")
                    achieved = measured_snr(clean, noisy_pcm, augmentation["mixture_gain"])
                    if abs(achieved - snr) > SNR_TOLERANCE_DB:
                        raise ValueError(f"PCM SNR outside tolerance: {achieved} vs {snr}")
                    augmentation["measured_snr_db"] = achieved
                    augmentation["source_audio_id"] = audio_id
                    noise_qc = check_wav(args.output / noisy_path)
                    qc.append({"audio_id": noisy_id, "measured_snr_db": achieved, **noise_qc})
                    manifest.write(json.dumps({**common, "audio_id": noisy_id,
                        "path": noisy_path, "duration_sec": noise_qc["duration_sec"],
                        "source_audio_id": audio_id, "augmentations": [augmentation],
                    }, ensure_ascii=False) + "\n")
                print(f"{audio_id}: clean + {len(config['NOISE_SNRS_DB'])} noise variants", flush=True)
    write_json(args.output / "audio_checks.json", qc)
    print(f"Completed: {len(qc)} WAV files in {args.output}")


if __name__ == "__main__":
    main()
