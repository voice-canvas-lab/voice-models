"""Shared input validation; standard library only, no model imports."""

import json
import math


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    voices = config.get("VOICES")
    snrs = config.get("NOISE_SNRS_DB")
    seed = config.get("SEED")
    if not isinstance(voices, list) or not voices or any(v not in ("aidar", "baya", "xenia") for v in voices):
        raise ValueError("VOICES must contain aidar, baya and/or xenia")
    if len(set(voices)) != len(voices):
        raise ValueError("Duplicate voices")
    if not isinstance(snrs, list) or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not -40 <= s <= 80 for s in snrs):
        raise ValueError("NOISE_SNRS_DB must be finite numbers between -40 and 80")
    if len(set(snrs)) != len(snrs):
        raise ValueError("Duplicate SNR levels")
    if len({f"{snr:g}" for snr in snrs}) != len(snrs):
        raise ValueError("SNR values produce duplicate file labels")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("SEED must be an integer in [0, 2**63)")
    return config


def validate_context(context, line_no):
    prefix = f"Line {line_no}: context"
    if not isinstance(context, dict):
        raise ValueError(f"{prefix} must be an object")
    if not {"objects", "selected_object_ids"} <= context.keys():
        raise ValueError(f"{prefix} requires objects and selected_object_ids")
    if not isinstance(context["objects"], list) or not isinstance(context["selected_object_ids"], list):
        raise ValueError(f"{prefix} objects and selected_object_ids must be arrays")
    object_ids = set()
    for obj in context["objects"]:
        if not isinstance(obj, dict):
            raise ValueError(f"{prefix} objects must contain objects")
        for field in ("object_id", "object_type", "name"):
            if not isinstance(obj.get(field), str) or not obj[field].strip():
                raise ValueError(f"{prefix} object has invalid {field}")
        if "attributes" in obj and not isinstance(obj["attributes"], dict):
            raise ValueError(f"{prefix} object attributes must be an object")
        if obj["object_id"] in object_ids:
            raise ValueError(f"{prefix} has duplicate object_id: {obj['object_id']}")
        object_ids.add(obj["object_id"])
    for selected_id in context["selected_object_ids"]:
        if not isinstance(selected_id, str) or not selected_id.strip():
            raise ValueError(f"{prefix} has invalid selected_object_id")
        if selected_id not in object_ids:
            raise ValueError(f"{prefix} selected_object_id not found: {selected_id}")


def load_samples(path):
    samples = []
    ids = set()
    groups = {}
    required = {"sample_id", "text", "context", "action_id", "args", "group_id", "split"}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no}: expected sample object")
        if not required <= row.keys():
            raise ValueError(f"Line {line_no}: missing fields")
        for field in ("sample_id", "text", "group_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Line {line_no}: invalid {field}")
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        validate_context(row["context"], line_no)
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        if not isinstance(row["args"], dict):
            raise ValueError("args must be an object")
        if row["action_id"] is not None and not isinstance(row["action_id"], str):
            raise ValueError("Invalid action_id")
        signature = (row["action_id"], row["args"], row["context"], row["split"])
        previous = groups.setdefault(row["group_id"], signature)
        if previous != signature:
            raise ValueError("Group has inconsistent labels or splits")
        ids.add(row["sample_id"])
        samples.append(row)
    if not samples:
        raise ValueError("No samples")
    return samples
