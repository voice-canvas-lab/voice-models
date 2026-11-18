"""Supported recognition settings, shared by preflight and ASR entrypoints."""

import json
import re

WHISPER_MODEL = "Systran/faster-whisper-small"
GIGAAM_MODEL = "ai-sage/GigaAM-v3"
WHISPER_DECODING = {
    "language": "ru", "task": "transcribe", "beam_size": 5,
    "temperature": 0.0, "condition_on_previous_text": False,
    "vad_filter": False, "word_timestamps": False,
    "initial_prompt": None, "hotwords": None,
}


def validate_asr(name, value):
    required = {"model", "revision", "device", "compute_type", "cpu_threads", "decoding"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"ASR.{name} requires exactly {sorted(required)}")
    expected_model = WHISPER_MODEL if name == "whisper" else GIGAAM_MODEL
    expected_type = "int8" if name == "whisper" else "float32"
    if value["model"] != expected_model or value["device"] != "cpu" or value["compute_type"] != expected_type:
        raise ValueError(f"ASR.{name}: supported model/device/type are {expected_model}/cpu/{expected_type}")
    if not isinstance(value["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise ValueError(f"ASR.{name}.revision must be an immutable 40-character commit")
    threads = value["cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError(f"ASR.{name}.cpu_threads must be in [1, 64]")
    decoding = value["decoding"]
    if not isinstance(decoding, dict):
        raise ValueError(f"ASR.{name}.decoding must be an object")
    if name == "gigaam":
        if decoding != {"method": "greedy_ctc"}:
            raise ValueError("GigaAM e2e_ctc supports its official greedy_ctc decoder only")
    else:
        if set(decoding) - WHISPER_DECODING.keys():
            raise ValueError("Unsupported Whisper decoding keys")
        merged = {**WHISPER_DECODING, **decoding}
        if merged["language"] != "ru" or merged["task"] != "transcribe" or merged["initial_prompt"] is not None or merged["hotwords"] is not None:
            raise ValueError("Whisper requires ru/transcribe and no prompt/hotwords")
        if isinstance(merged["beam_size"], bool) or not isinstance(merged["beam_size"], int) or not 1 <= merged["beam_size"] <= 20:
            raise ValueError("Invalid Whisper beam_size")
        if isinstance(merged["temperature"], bool) or not isinstance(merged["temperature"], (int, float)) or not 0 <= merged["temperature"] <= 1:
            raise ValueError("Invalid Whisper temperature")
        for key in ("condition_on_previous_text", "vad_filter", "word_timestamps"):
            if not isinstance(merged[key], bool):
                raise ValueError(f"Whisper {key} must be boolean")
    return value


def from_file(path, name):
    config = json.loads(path.read_text(encoding="utf-8"))
    return validate_asr(name, config["ASR"][name])


def validate_tts(value):
    required = {"model", "model_sha256", "cpu_threads", "put_accent", "put_yo"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"TTS requires exactly {sorted(required)}")
    if value["model"] != "v5_ru" or not isinstance(value["model_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["model_sha256"]):
        raise ValueError("TTS requires v5_ru and its SHA256 revision")
    threads = value["cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("Invalid TTS cpu_threads")
    if any(not isinstance(value[key], bool) for key in ("put_accent", "put_yo")):
        raise ValueError("TTS accent/yo settings must be boolean")
    return value
