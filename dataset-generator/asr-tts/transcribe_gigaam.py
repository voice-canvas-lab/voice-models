"""Official HF GigaAM-v3 e2e_ctc, CPU: manifest -> predictions and timings."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import imageio_ffmpeg
import soundfile as sf
import torch
from huggingface_hub import HfApi, snapshot_download

os.environ.setdefault("HF_MODULES_CACHE", str(Path(".cache/gigaam/modules").resolve()))
from transformers import AutoModel
from recognition_config import from_file


MODEL = "ai-sage/GigaAM-v3"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def transcribe(model, path):
    with torch.inference_mode():
        text = model.transcribe(str(path))
    if not isinstance(text, str):
        raise TypeError("Expected complete text from pinned e2e_ctc implementation")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="cec030b4c4f35d928e4a9044a3bdb29ebd499fac")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/gigaam"))
    parser.add_argument("--config", type=Path, help="Pipeline config; overrides model/revision/threads")
    args = parser.parse_args()
    run_config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    recognition = from_file(args.config, "gigaam") if args.config else None
    model_id = recognition["model"] if recognition else MODEL
    requested_revision = recognition["revision"] if recognition else args.revision
    threads = recognition["cpu_threads"] if recognition else 4
    seed = run_config.get("SEED", 42)
    root = args.dataset.resolve()
    manifest_path = root / "audio_manifest.jsonl"
    rows, ids = [], set()
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        for field in ("audio_id", "sample_id", "path"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Invalid {field}")
        if row["audio_id"] in ids:
            raise ValueError("Duplicate audio_id")
        path = (root / row["path"]).resolve()
        if Path(row["path"]).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Audio path must be relative to dataset root")
        info = sf.info(path)
        if (info.format, info.subtype, info.channels, info.samplerate) != ("WAV", "PCM_16", 1, 16000):
            raise ValueError(f"Unexpected WAV: {path}")
        duration = info.frames / info.samplerate
        if not 0 < duration <= 25:
            raise ValueError("GigaAM short-form requires 0 < duration <= 25 seconds")
        rows.append((row, path, duration))
        ids.add(row["audio_id"])
    if not rows:
        raise ValueError("Empty manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Official implementation invokes `ffmpeg`. Use a project-local bundled binary.
    ffmpeg_dir = (args.cache_dir / "ffmpeg").resolve()
    ffmpeg_dir.mkdir(exist_ok=True)
    ffmpeg = ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
    ffmpeg_version = subprocess.check_output([str(ffmpeg), "-version"], text=True).splitlines()[0]
    revision = HfApi().model_info(model_id, revision=requested_revision).sha
    model_dir = snapshot_download(
        model_id, revision=revision, cache_dir=str(args.cache_dir / "hub"),
        allow_patterns=["*.json", "*.py", "pytorch_model.bin", "tokenizer.model"],
    )
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    cfg = config["cfg"]["model"]["cfg"]
    if cfg["model_name"] != "v3_e2e_ctc" or cfg["model_class"] != "ctc":
        raise ValueError("Refusing to substitute a model other than v3_e2e_ctc")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to("cpu").float().eval()
    print("Loaded v3_e2e_ctc on CPU float32; full warmup (not timed).", flush=True)
    transcribe(model, rows[0][1])
    settings = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id, "variant": "e2e_ctc", "model_name": cfg["model_name"],
        "requested_revision": requested_revision, "resolved_revision": revision,
        "device": "cpu", "compute_type": "float32", "cpu_threads": threads, "seed": seed,
        "decoding": cfg["decoding"], "model_config": config,
        "input": "WAV path only; no reference, context, actions or args",
        "hardware": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                     "os": platform.platform(), "machine": platform.machine()},
        "python": platform.python_version(), "ffmpeg": ffmpeg_version,
        "dependencies": sorted([{"name": d.metadata["Name"], "version": d.version}
                                for d in importlib.metadata.distributions()], key=lambda d: d["name"].lower()),
        "manifest_sha256": digest(manifest_path), "script_sha256": digest(Path(__file__)),
        "warmup": {"runs": 1, "audio_id": rows[0][0]["audio_id"]},
        "timing": "perf_counter: FFmpeg reading, features, inference and complete text; excludes loading, warmup and output I/O",
        "status": "running",
    }
    settings_path = args.output / "asr_settings.json"
    with settings_path.open("x", encoding="utf-8") as stream:
        json.dump(settings, stream, ensure_ascii=False, indent=2)
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
            (args.output / "timings.jsonl").open("x", encoding="utf-8") as timings:
        for row, path, duration in rows:
            start = perf_counter()
            text = transcribe(model, path)
            elapsed = perf_counter() - start
            prediction = {"sample_id": row["sample_id"], "audio_id": row["audio_id"], "text": text}
            timing = {"sample_id": row["sample_id"], "audio_id": row["audio_id"],
                      "duration_sec": duration, "elapsed_sec": elapsed, "rtf": elapsed / duration,
                      "audio_sha256": digest(path)}
            predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            timings.write(json.dumps(timing) + "\n")
            predictions.flush()
            timings.flush()
            print(f"{row['audio_id']}: {text!r}, {elapsed:.3f}s, RTF={elapsed / duration:.3f}", flush=True)
    settings["status"] = "completed"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
