"""Corpus metrics by voice and noise condition from validated ASR evaluations."""

import argparse
import json
from pathlib import Path

from evaluate import RULES, aggregate, digest, read_index
from fragment_metrics import MATCH_RULES, report_lines, summarize


def condition(row):
    return "clean" if not row["augmentations"] else f"snr{row['augmentations'][0]['snr_db']:g}"


def group_metrics(records, manifest, samples):
    groups = {"all": list(records.values())}
    for aid, row in records.items():
        audio = manifest[aid]
        if row["sample_id"] != audio["sample_id"] or audio["sample_id"] not in samples:
            raise ValueError("Evaluation sample_id differs")
        action = samples[audio["sample_id"]]["action_id"]
        split = samples[audio["sample_id"]]["split"]
        for key in (f"action_id={action}", f"voice={audio['voice']}",
                    f"condition={condition(audio)}", f"split={split}"):
            groups.setdefault(key, []).append(row)
    result = {}
    for key, rows in groups.items():
        ids = {row["sample_id"] for row in rows}
        result[key] = {**aggregate(rows), "unique_samples": len(ids),
                       "unique_group_ids": len({samples[sid]["group_id"] for sid in ids})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True, help="Evaluation directory")
    parser.add_argument("--gigaam", type=Path, required=True, help="Evaluation directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--whisper-fragments", type=Path)
    parser.add_argument("--gigaam-fragments", type=Path)
    args = parser.parse_args()
    if bool(args.whisper_fragments) != bool(args.gigaam_fragments):
        parser.error("Provide both fragment evaluations or neither")
    manifest = read_index(args.dataset / "audio_manifest.jsonl", "audio_id")
    samples = read_index(args.dataset / "samples.jsonl", "sample_id")
    config = json.loads((args.dataset / "config.json").read_text(encoding="utf-8"))
    qc = json.loads((args.dataset / "dataset_checks.json").read_text(encoding="utf-8"))
    if not qc["passed"] or qc["files"] != len(manifest):
        raise ValueError("Dataset audit failed or count differs")
    lines = ["# Multi-voice ASR technical comparison", "",
             ("Artificial white Gaussian RNG noise, NOT a recording of a real acoustic environment."
              if qc["noisy"] else "No noise configured for this run; all WAV files are clean."),
             "Power is mean(x²) across ALL decoded clean PCM samples, including silence; no VAD or DC removal.",
             "Noise is scaled to target power; any peak protection scales the whole mixture, preserving SNR.",
             "Achieved SNR is measured from decoded noisy PCM versus gain × clean PCM; residual includes quantization.",
             f"Files: {qc['files']} ({qc['clean']} clean, {qc['noisy']} noisy); WAV mono 16 kHz PCM_16.",
             f"QC: passed, clipped files={qc['clipped_files']}, max peak={qc['max_peak']:.6f}, max SNR error={qc['max_snr_error_db']:.6f} dB.",
             f"Seed {config['SEED']}; each noise file has its own stable PCG64 seed and clean source_audio_id.",
             f"Samples: {len(samples)}; paraphrase groups: {len({row['group_id'] for row in samples.values()})}; splits: {sorted({row['split'] for row in samples.values()})}.",
             "Whisper CPU INT8 and GigaAM-v3 e2e_ctc CPU float32; each runner records decoding, thread settings and its full warmup separately.",
             "These separate runs and this tiny synthetic set are a technical check, not a general quality or speed benchmark.",
             "", RULES, "", "## Corpus metrics", "",
             "| Model | Group | WAV | Unique samples | Unique group_id | WER | CER | RTF | Word errors/ref | Char errors/ref |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    all_groups, all_records, fragment_summaries, prediction_hashes = {}, {}, {}, {}
    for model, folder in (("Whisper", args.whisper), ("GigaAM-v3 e2e_ctc", args.gigaam)):
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        prediction_hashes[model] = summary["input_sha256"]["predictions"]
        for name, file in (("manifest", "audio_manifest.jsonl"), ("samples", "samples.jsonl")):
            if summary["input_sha256"][name] != digest(args.dataset / file):
                raise ValueError("Evaluation uses different dataset")
        records = read_index(folder / "record_metrics.jsonl", "audio_id")
        if records.keys() != manifest.keys():
            raise ValueError("Evaluation IDs differ")
        all_groups[model] = group_metrics(records, manifest, samples)
        all_records[model] = list(records.values())
        for key, totals in all_groups[model].items():
            wer = "undefined" if totals["wer"] is None else f"{totals['wer']:.2%}"
            cer = "undefined" if totals["cer"] is None else f"{totals['cer']:.2%}"
            lines.append(f"| {model} | {key} | {totals['records']} | {totals['unique_samples']} | {totals['unique_group_ids']} | {wer} | {cer} | {totals['rtf']:.3f} | {totals['word_errors']}/{totals['reference_words']} | {totals['character_errors']}/{totals['reference_characters']} |")
    lines += ["", "## Lexical fragment retention", "", MATCH_RULES, ""]
    for model, directory in (("Whisper", args.whisper_fragments), ("GigaAM-v3 e2e_ctc", args.gigaam_fragments)):
        retention = summarize([])
        if directory:
            retention = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            for name, file in (("manifest", "audio_manifest.jsonl"), ("samples", "samples.jsonl")):
                if retention["input_sha256"][name] != digest(args.dataset / file):
                    raise ValueError("Fragment evaluation uses different dataset")
            if retention["input_sha256"]["predictions"] != prediction_hashes[model]:
                raise ValueError("Fragment evaluation uses different predictions")
        fragment_summaries[model] = retention
        lines.extend([f"### {model}", "", *report_lines(retention), ""])
        if directory:
            lines.append(f"Detailed fragment errors: {directory / 'report.md'}")
    if args.whisper_fragments and len({s["input_sha256"]["annotations"] for s in fragment_summaries.values()}) != 1:
        raise ValueError("Models were evaluated with different annotations")
    lines += ["", "## Error examples (first 12 per model; all records in JSON)", ""]
    for model, rows in all_records.items():
        errors = [row for row in rows if row["word_errors"] or row["character_errors"]]
        lines += [f"### {model}: {len(errors)} records with errors", ""]
        for row in errors[:12]:
            audio = manifest[row["audio_id"]]
            lines += [f"- {row['audio_id']} ({audio['voice']}, {condition(audio)}):",
                      f"  Reference: {json.dumps(row['reference_text'], ensure_ascii=False)}",
                      f"  Prediction: {json.dumps(row['prediction_text'], ensure_ascii=False)}",
                      f"  Word S/D/I: {row['word_substitutions']}/{row['word_deletions']}/{row['word_insertions']}"]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({"models": all_groups, "audio_checks": qc, "normalization": RULES,
                   "unique_samples": len(samples), "unique_group_ids": len({s["group_id"] for s in samples.values()}),
                   "fragment_retention": fragment_summaries}, stream, ensure_ascii=False, indent=2)
    with (args.output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    with (args.output / "records.json").open("x", encoding="utf-8") as stream:
        json.dump(all_records, stream, ensure_ascii=False, indent=2)
    print(f"Comparison saved: {args.output}")


if __name__ == "__main__":
    main()
