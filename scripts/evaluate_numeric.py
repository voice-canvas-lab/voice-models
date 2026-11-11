"""Additional numeric-equivalent WER/CER; existing evaluation is read-only."""

import argparse
import json
from pathlib import Path

from evaluate import RULES, aggregate, check_links, digest, read_index, score
from numeric_equivalence import VERSION, canonicalize

POLICY = {
    "independence": "canonicalize(text) is applied separately to reference and hypothesis; it has no other input and does not repair predictions.",
    "supported": "Russian month-name calendar dates; versions after explicit версия; clock units hour/minute[/second] or valid compact clocks in explicit время context; known opaque ID prefixes.",
    "zero_policy": "Every explicit zero and component width is retained. No inferred zero padding. Thus 06 differs from 6, 09 differs from 9, ID 000001 differs from 1; explicit ноль один equals 01.",
    "boundaries": "Typed tokens numdate/numtime/numversion/numid with ordered named components and end markers. No reordering, merging or filling missing components.",
    "unknowns": "Unsupported/ambiguous numeric expressions remain literal and receive flags. No ISO/slash date inference, bare numbers, naked compact time inference or guessed splitting of 17:4029.",
    "metric_units": "Additional WER/CER include typed canonical tokens and therefore have different denominators from baseline. They are a supplemental encoding-sensitive diagnostic, not a replacement for baseline ASR quality metrics.",
    "base_normalization": "Existing evaluate.score/normalize is applied after this separate stage, unchanged. Unknown spans are preserved before that existing punctuation normalization.",
}


def evaluate(dataset, results, baseline, output):
    paths = {"samples": dataset / "samples.jsonl", "manifest": dataset / "audio_manifest.jsonl",
             "predictions": results / "predictions.jsonl", "timings": results / "timings.jsonl"}
    indexes = {name: read_index(path, "sample_id" if name == "samples" else "audio_id") for name, path in paths.items()}
    check_links(**indexes)
    old = read_index(baseline / "record_metrics.jsonl", "audio_id")
    assert old.keys() == indexes["manifest"].keys(), "Baseline IDs differ"
    original_summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    for name, path in paths.items():
        assert original_summary["input_sha256"][name] == digest(path), f"Baseline input differs: {name}"
    input_hashes = {str(path): digest(path) for path in [*paths.values(), baseline / "record_metrics.jsonl", baseline / "summary.json"]}
    records, extra_records = [], []
    for aid, audio in indexes["manifest"].items():
        ref = indexes["samples"][audio["sample_id"]]["text"]
        hyp = indexes["predictions"][aid]["text"]
        original_score = score(ref, hyp)
        assert all(old[aid][key] == value for key, value in original_score.items()), "Baseline scoring changed"
        # These calls are independent; neither receives the opposite side.
        ref_numeric = canonicalize(ref)
        hyp_numeric = canonicalize(hyp)
        extra = score(ref_numeric["transformed_text"], hyp_numeric["transformed_text"])
        timing = indexes["timings"][aid]
        extra_records.append({**extra, "elapsed_sec": timing["elapsed_sec"], "duration_sec": timing["duration_sec"]})
        records.append({"sample_id": audio["sample_id"], "audio_id": aid,
                        "baseline": old[aid], "numeric_equivalent": extra,
                        "reference_transform": ref_numeric, "hypothesis_transform": hyp_numeric})
    before_summary = aggregate(list(old.values()))
    after_summary = aggregate(extra_records)
    summary = {"rules_version": VERSION, "policy": POLICY, "normalization": RULES,
               "baseline_metrics_verified_unchanged": True, "baseline": before_summary,
               "numeric_equivalent": after_summary,
               "records_with_reference_flags": sum(r["reference_transform"]["needs_review"] for r in records),
               "records_with_hypothesis_flags": sum(r["hypothesis_transform"]["needs_review"] for r in records),
               "asr_run": False, "input_sha256": input_hashes,
               "numeric_rules_sha256": digest(Path(__file__).with_name("numeric_equivalence.py")),
               "evaluation_script_sha256": digest(Path(__file__)),
               "baseline_script_sha256": digest(Path(__file__).with_name("evaluate.py"))}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "record_metrics.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Дополнительная оценка числовых выражений", "", f"Правила: {VERSION}.",
             "Существующие WER/CER, нормализация, эталоны и гипотезы не изменены. ASR не запускалась.",
             "Один и тот же canonicalize(text) независимо применяется к каждой стороне сравнения.",
             "Компоненты помечены типами и сохраняют порядок/ведущие нули. Неизвестное не исправляется.",
             "Дополнительные WER/CER имеют другие знаменатели из-за канонических токенов. Это отдельный диагностический показатель.", "",
             "| sample_id | Исходный WER | Исходный CER | Доп. WER | Доп. CER |", "|---|---:|---:|---:|---:|"]
    for row in records:
        a, b = row["baseline"], row["numeric_equivalent"]
        lines.append(f"| {row['sample_id']} | {a['wer']:.2%} | {a['cer']:.2%} | {b['wer']:.2%} | {b['cer']:.2%} |")
    for row in records:
        lines += ["", "## " + row["sample_id"], "", "audio_id: " + row["audio_id"]]
        for name, key in (("Эталон", "reference_transform"), ("Гипотеза", "hypothesis_transform")):
            transformation = row[key]
            lines += ["", name + " до: " + transformation["original_text"], "",
                      name + " после: " + transformation["transformed_text"], "",
                      "Флаги: " + json.dumps(transformation["flags"], ensure_ascii=False)]
        a, b = row["baseline"], row["numeric_equivalent"]
        lines += ["", f"Слова: {a['word_errors']}/{a['reference_words']} → {b['word_errors']}/{b['reference_words']}; символы: {a['character_errors']}/{a['reference_characters']} → {b['character_errors']}/{b['reference_characters']}."]
    lines += ["", "## Границы правил", ""]
    lines.extend(f"- {name}: {value}" for name, value in POLICY.items())
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert all(digest(Path(path)) == checksum for path, checksum in input_hashes.items()), "Inputs changed during evaluation"
    print(json.dumps({"records": len(records), "baseline_wer": before_summary["wer"],
                      "additional_wer": after_summary["wer"], "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "results", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.results, args.baseline, args.output)


if __name__ == "__main__":
    main()
