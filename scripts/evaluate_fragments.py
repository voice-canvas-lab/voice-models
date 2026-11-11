"""Evaluate manually annotated lexical fragments only after ASR has completed."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from evaluate import RULES, check_prediction_links, digest, normalize, read_index
from fragment_metrics import MATCH_RULES, load_annotations, report_lines, score_fragments, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples_path = args.dataset / "samples.jsonl"
    manifest_path = args.dataset / "audio_manifest.jsonl"
    predictions_path = args.results / "predictions.jsonl"
    samples = read_index(samples_path, "sample_id")
    manifest = read_index(manifest_path, "audio_id")
    predictions = read_index(predictions_path, "audio_id")
    check_prediction_links(samples, manifest, predictions)
    annotations = load_annotations(args.annotations, samples)
    records = []
    for aid, audio in manifest.items():
        sid = audio["sample_id"]
        reference, prediction = samples[sid]["text"], predictions[aid]["text"]
        records.append({"sample_id": sid, "audio_id": aid, "reference_text": reference,
                        "prediction_text": prediction,
                        "normalized_reference": normalize(reference), "normalized_prediction": normalize(prediction),
                        "fragments": score_fragments(reference, prediction, annotations.get(sid, []))})
    summary = summarize(records)
    summary.update(normalization=RULES, jiwer_version=importlib.metadata.version("jiwer"),
                   script_sha256=digest(Path(__file__)), matching_code_sha256=digest(Path(__file__).with_name("fragment_metrics.py")),
                   unique_annotated_samples=sum(bool(fs) for fs in annotations.values()),
                   unique_reference_fragments=sum(len(fs) for fs in annotations.values()),
                   input_sha256={"samples": digest(samples_path), "manifest": digest(manifest_path),
                                 "predictions": digest(predictions_path),
                                 "annotations": digest(args.annotations) if args.annotations else None})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "record_fragments.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    lines = ["# Lexical fragment retention", "", MATCH_RULES, "", *report_lines(summary), "",
             "Each annotated fragment is checked separately for every audio version. Unannotated categories are not evaluated.",
             "This checks lexical preservation, not correct intent, negation scope or final corrected values.", "", "## Errors", ""]
    errors = 0
    for row in records:
        for fragment in row["fragments"]:
            if not fragment["preserved"]:
                errors += 1
                lines.extend([f"- {row['audio_id']} / {row['sample_id']} / {fragment['fragment_id']} ({fragment['category']}): {fragment['reason']}",
                              f"  Reference: {json.dumps(row['reference_text'], ensure_ascii=False)}",
                              f"  Prediction: {json.dumps(row['prediction_text'], ensure_ascii=False)}",
                              f"  Fragment: {json.dumps(fragment, ensure_ascii=False)}"])
    if not errors:
        lines.append("No errors among evaluated fragments (see checked counts; no annotations means not evaluated).")
    with (args.output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(summary["categories"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
