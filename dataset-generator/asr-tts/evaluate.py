"""Evaluate ASR predictions against a dataset snapshot without changing inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import unicodedata
from pathlib import Path

import jiwer


RULES = (
    "Unicode NFC → lowercase → ё to е → Unicode punctuation (category P*) "
    "replaced by spaces → whitespace collapsed. Words, numbers, negations and "
    "self-corrections are retained. CER includes spaces in normalized strings. "
    "Empty normalized references are kept: hypothesis tokens count as insertions; "
    "a rate with zero reference denominator is null (undefined), even for two "
    "empty strings. Their errors still contribute to corpus totals. Corpus rates "
    "are summed errors / summed reference units, never mean per-record rates."
)


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower().replace("ё", "е")
    return " ".join("".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    ).split())


def read_index(path, key):
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected object")
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{path}:{number}: invalid {key}")
        if identifier in result:
            raise ValueError(f"{path}:{number}: duplicate {key}={identifier}")
        result[identifier] = row
    if not result:
        raise ValueError(f"Empty file: {path}")
    return result


def check_prediction_links(samples, manifest, predictions):
    missing = sorted(manifest.keys() - predictions.keys())
    extra = sorted(predictions.keys() - manifest.keys())
    if missing or extra:
        raise ValueError(f"predictions: missing audio_id={missing}; extra audio_id={extra}")
    for audio_id, audio in manifest.items():
        sample_id = audio.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in samples:
            raise ValueError(f"{audio_id}: unknown sample_id={sample_id}")
        if predictions[audio_id].get("sample_id") != sample_id:
            raise ValueError(f"{audio_id}: prediction sample_id mismatch")
        if not isinstance(samples[sample_id].get("text"), str):
            raise ValueError(f"{sample_id}: reference text must be a string")
        if not isinstance(predictions[audio_id].get("text"), str):
            raise ValueError(f"{audio_id}: prediction text must be a string")


def check_links(samples, manifest, predictions, timings):
    check_prediction_links(samples, manifest, predictions)
    missing = sorted(manifest.keys() - timings.keys())
    extra = sorted(timings.keys() - manifest.keys())
    if missing or extra:
        raise ValueError(f"timings: missing audio_id={missing}; extra audio_id={extra}")
    for audio_id, audio in manifest.items():
        if timings[audio_id].get("sample_id") != audio["sample_id"]:
            raise ValueError(f"{audio_id}: timing sample_id mismatch")
        timing = timings[audio_id]
        for field in ("elapsed_sec", "duration_sec"):
            value = timing.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{audio_id}: invalid {field}")
            if value < 0 or (field == "duration_sec" and value == 0):
                raise ValueError(f"{audio_id}: invalid {field}")
        if "duration_sec" in audio:
            value = audio["duration_sec"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{audio_id}: invalid manifest duration")
            if not math.isclose(value, timing["duration_sec"], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{audio_id}: duration mismatch")


def rate(errors, denominator):
    return errors / denominator if denominator else None


def score(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    word_errors = words.substitutions + words.deletions + words.insertions
    char_errors = chars.substitutions + chars.deletions + chars.insertions
    return {
        "reference_text": reference, "prediction_text": hypothesis,
        "normalized_reference": ref, "normalized_prediction": hyp,
        "empty_reference": not ref,
        "reference_words": len(ref.split()), "reference_characters": len(ref),
        "word_substitutions": words.substitutions, "word_deletions": words.deletions,
        "word_insertions": words.insertions, "word_errors": word_errors,
        "character_substitutions": chars.substitutions, "character_deletions": chars.deletions,
        "character_insertions": chars.insertions, "character_errors": char_errors,
        "wer": rate(word_errors, len(ref.split())), "cer": rate(char_errors, len(ref)),
    }


def aggregate(records):
    fields = ("reference_words", "reference_characters", "word_substitutions",
              "word_deletions", "word_insertions", "word_errors", "character_substitutions",
              "character_deletions", "character_insertions", "character_errors",
              "elapsed_sec", "duration_sec")
    totals = {field: sum(row[field] for row in records) for field in fields}
    return {
        "records": len(records), "empty_references": sum(row["empty_reference"] for row in records),
        **totals, "wer": rate(totals["word_errors"], totals["reference_words"]),
        "cer": rate(totals["character_errors"], totals["reference_characters"]),
        "rtf": rate(totals["elapsed_sec"], totals["duration_sec"]),
    }


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display(value):
    return "undefined" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"samples": args.dataset / "samples.jsonl",
             "manifest": args.dataset / "audio_manifest.jsonl",
             "predictions": args.results / "predictions.jsonl",
             "timings": args.results / "timings.jsonl"}
    indexes = {name: read_index(path, "sample_id" if name == "samples" else "audio_id")
               for name, path in paths.items()}
    check_links(**indexes)
    records = []
    for audio_id, audio in indexes["manifest"].items():
        sample_id = audio["sample_id"]
        timing = indexes["timings"][audio_id]
        records.append({
            "audio_id": audio_id, "sample_id": sample_id,
            **score(indexes["samples"][sample_id]["text"], indexes["predictions"][audio_id]["text"]),
            "elapsed_sec": timing["elapsed_sec"], "duration_sec": timing["duration_sec"],
            "rtf": timing["elapsed_sec"] / timing["duration_sec"],
        })
    totals = aggregate(records)
    summary = {**totals, "normalization": RULES,
               "jiwer_version": importlib.metadata.version("jiwer"),
               "input_sha256": {name: digest(path) for name, path in paths.items()},
               "script_sha256": digest(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "record_metrics.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    report = ["# ASR evaluation", "", RULES, "",
              f"Records: {totals['records']}; empty normalized references: {totals['empty_references']}.",
              f"Corpus WER: {display(totals['wer'])} = {totals['word_errors']} / {totals['reference_words']}.",
              f"Corpus CER: {display(totals['cer'])} = {totals['character_errors']} / {totals['reference_characters']}.",
              f"Overall RTF: {display(totals['rtf'])} = {totals['elapsed_sec']:.6f} s / {totals['duration_sec']:.6f} s.",
              "", "## Errors and empty references", ""]
    errors = [row for row in records if row["word_errors"] or row["character_errors"] or row["empty_reference"]]
    if not errors:
        report.append("No errors after normalization.")
    for row in errors:
        report.extend([
            f"### {row['audio_id']} ({row['sample_id']})", "",
            f"- Reference: {json.dumps(row['reference_text'], ensure_ascii=False)}",
            f"- Prediction: {json.dumps(row['prediction_text'], ensure_ascii=False)}",
            f"- Normalized reference: {json.dumps(row['normalized_reference'], ensure_ascii=False)}",
            f"- Normalized prediction: {json.dumps(row['normalized_prediction'], ensure_ascii=False)}",
            f"- Word substitutions/deletions/insertions: {row['word_substitutions']}/{row['word_deletions']}/{row['word_insertions']}",
            f"- WER: {display(row['wer'])}; CER: {display(row['cer'])}", "",
        ])
    with (args.output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
