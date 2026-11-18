"""Audit the complete configured audio set, including actual PCM SNR and links."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_noise import SNR_TOLERANCE_DB, measured_snr, mix_noise, power, record_seed


def unique_rows(path, key):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    index = {row[key]: row for row in rows}
    if len(index) != len(rows) or not rows:
        raise ValueError(f"Empty file or duplicate {key}: {path}")
    return index


def audit(root):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    samples = unique_rows(root / "samples.jsonl", "sample_id")
    manifest = unique_rows(root / "audio_manifest.jsonl", "audio_id")
    expected_clean = len(samples) * len(config["VOICES"])
    expected_total = expected_clean * (1 + len(config["NOISE_SNRS_DB"]))
    if len(manifest) != expected_total:
        raise ValueError("Unexpected manifest count")
    actual_paths = set()
    seen_combinations = set()
    deviations, measured, peaks = [], [], []
    clean_count = 0
    conditions = Counter()
    groups = {}
    for sample in samples.values():
        split = groups.setdefault(sample["group_id"], sample["split"])
        if split != sample["split"]:
            raise ValueError("Paraphrase group crosses splits")
    for audio_id, row in manifest.items():
        sample = samples[row["sample_id"]]
        if row["voice"] not in config["VOICES"] or row["split"] != sample["split"] or row["seed"] != config["SEED"]:
            raise ValueError("Voice, split or seed mismatch")
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()) or path in actual_paths:
            raise ValueError("Invalid or duplicate path")
        actual_paths.add(path)
        info = sf.info(path)
        pcm, sr = sf.read(path, dtype="float64")
        if (info.format, info.subtype, info.channels, sr) != ("WAV", "PCM_16", 1, 16000):
            raise ValueError("Invalid WAV format")
        if not pcm.size or not np.isfinite(pcm).all() or power(pcm) < 1e-10:
            raise ValueError("Empty or silent/nonfinite WAV")
        if not 0.2 <= len(pcm) / sr <= 30 or abs(row["duration_sec"] - len(pcm) / sr) > 1e-9:
            raise ValueError("Invalid duration")
        peak = float(np.max(np.abs(pcm)))
        if peak >= 0.999:
            raise ValueError("Clipping detected")
        peaks.append(peak)
        if not row["augmentations"]:
            if row.get("source_audio_id") is not None:
                raise ValueError("Clean record has a source")
            snr = None
            clean_count += 1
            conditions["clean"] += 1
        else:
            if len(row["augmentations"]) != 1:
                raise ValueError("Unexpected augmentations")
            aug = row["augmentations"][0]
            snr = aug["snr_db"]
            if snr not in config["NOISE_SNRS_DB"] or aug["noise_type"] != "artificial_white_gaussian":
                raise ValueError("Unexpected noise")
            source = manifest[row["source_audio_id"]]
            if source["augmentations"] or source["audio_id"] != aug["source_audio_id"]:
                raise ValueError("Noise must reference a clean original")
            for field in ("sample_id", "voice", "split"):
                if source[field] != row[field]:
                    raise ValueError("Source relationship mismatch")
            clean, _ = sf.read(root / source["path"], dtype="float64")
            seed = record_seed(config["SEED"], row["sample_id"], row["voice"], snr)
            if seed != aug["seed"]:
                raise ValueError("Noise seed mismatch")
            reconstructed, parameters = mix_noise(clean, snr, seed)
            for field in ("noise_scale", "mixture_gain", "sample_count"):
                if parameters[field] != aug[field]:
                    raise ValueError("Noise parameter mismatch")
            buffer = io.BytesIO()
            sf.write(buffer, reconstructed, sr, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            reconstructed_pcm, _ = sf.read(buffer, dtype="float64")
            if not np.array_equal(pcm, reconstructed_pcm):
                raise ValueError("Stored noise WAV does not match seed/settings")
            achieved = measured_snr(clean, pcm, aug["mixture_gain"])
            deviation = abs(achieved - snr)
            if deviation > SNR_TOLERANCE_DB or abs(achieved - aug["measured_snr_db"]) > 1e-9:
                raise ValueError("PCM SNR mismatch")
            deviations.append(deviation)
            measured.append({"audio_id": audio_id, "target_snr_db": snr, "measured_snr_db": achieved})
            conditions[f"snr{snr:g}"] += 1
        combo = (row["sample_id"], row["voice"], snr)
        if combo in seen_combinations:
            raise ValueError("Duplicate sample/voice/condition")
        seen_combinations.add(combo)
    expected = {(sid, voice, snr) for sid in samples for voice in config["VOICES"]
                for snr in [None, *config["NOISE_SNRS_DB"]]}
    if seen_combinations != expected or clean_count != expected_clean:
        raise ValueError("Incomplete sample/voice/condition coverage")
    if {p.resolve() for p in (root / "audio").rglob("*.wav")} != actual_paths:
        raise ValueError("Extra or missing WAV files")
    return {"passed": True, "files": len(manifest), "clean": clean_count,
            "noisy": len(manifest) - clean_count, "conditions": dict(conditions),
            "max_peak": max(peaks), "clipped_files": 0,
            "max_snr_error_db": max(deviations, default=0),
            "snr_tolerance_db": SNR_TOLERANCE_DB, "measured_snr": measured,
            "noise_source": "Artificial white Gaussian RNG noise, NOT a real acoustic environment recording",
            "noise_pcm_reconstruction": "bit-exact for every noisy file"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    args = parser.parse_args()
    result = audit(args.dataset)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "measured_snr"}, indent=2))


if __name__ == "__main__":
    main()
