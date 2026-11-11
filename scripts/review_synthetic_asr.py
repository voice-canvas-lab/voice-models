"""Audit paired ASR inputs and flag synthetic speech for human listening.

Heuristics identify candidates, never prove TTS omissions or alter references.
No models are loaded. Every manifest row remains in the reported metrics.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from compare_conditions import condition, group_metrics
from evaluate import RULES, read_index


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suspicion_reasons(reference, duration, metrics):
    reasons = []
    if duration > 25:
        reasons.append("duration_over_gigaam_25_seconds")
    chars = len(reference["normalized_reference"])
    if chars / duration > 25:
        reasons.append("reference_over_25_characters_per_audio_second")
    if reference["reference_words"] / duration > 5:
        reasons.append("reference_over_5_words_per_audio_second")
    if re.search(r"\d", reference["reference_text"]):
        if all(not re.search(r"\d", row["prediction_text"]) for row in metrics):
            reasons.append("reference_digits_but_neither_asr_returns_digits")
    if all(row["word_deletions"] / max(row["reference_words"], 1) >= 0.25
           and len(row["normalized_prediction"]) / max(chars, 1) < 0.75
           for row in metrics):
        reasons.append("both_asr_delete_at_least_quarter_words_and_shorter_text")
    return reasons


def review(dataset, results, output, source):
    # Reserve a new output folder; previous reports are never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    samples = read_index(dataset / "samples.jsonl", "sample_id")
    original = read_index(source, "sample_id")
    assert all(row == original[sid] for sid, row in samples.items()), "Sample content changed"
    manifest = read_index(dataset / "audio_manifest.jsonl", "audio_id")
    assert {row["sample_id"] for row in manifest.values()} == set(samples)
    model_records, grouped = {}, {}
    fingerprints = {aid: sha256(dataset / row["path"]) for aid, row in manifest.items()}
    technical = {}
    for model in ("whisper", "gigaam"):
        records = read_index(results / f"evaluation-{model}/record_metrics.jsonl", "audio_id")
        predictions = read_index(results / model / "predictions.jsonl", "audio_id")
        timings = read_index(results / model / "timings.jsonl", "audio_id")
        settings = json.loads((results / model / "asr_settings.json").read_text(encoding="utf-8"))
        evaluation = json.loads((results / f"evaluation-{model}/summary.json").read_text(encoding="utf-8"))
        assert records.keys() == predictions.keys() == timings.keys() == manifest.keys()
        assert settings["status"] == "completed"
        assert settings["manifest_sha256"] == sha256(dataset / "audio_manifest.jsonl")
        for key, path in (("samples", dataset / "samples.jsonl"),
                          ("manifest", dataset / "audio_manifest.jsonl"),
                          ("predictions", results / model / "predictions.jsonl"),
                          ("timings", results / model / "timings.jsonl")):
            assert evaluation["input_sha256"][key] == sha256(path), (model, key)
        for aid, audio in manifest.items():
            assert predictions[aid]["sample_id"] == timings[aid]["sample_id"] == audio["sample_id"]
            assert records[aid]["reference_text"] == samples[audio["sample_id"]]["text"]
            assert timings[aid]["audio_sha256"] == fingerprints[aid], (model, aid)
        model_records[model] = records
        grouped[model] = group_metrics(records, manifest, samples)
        technical[model] = {"successful": len(records), "failed_or_missing": 0,
                            "empty_predictions": sum(not p["text"].strip() for p in predictions.values()),
                            "records_with_word_or_character_errors": sum(bool(r["word_errors"] or r["character_errors"]) for r in records.values()),
                            "manifest_and_all_audio_hashes_match": True}
    suspicious = []
    for aid, audio in manifest.items():
        rows = [model_records[model][aid] for model in ("whisper", "gigaam")]
        reasons = suspicion_reasons(rows[0], audio["duration_sec"], rows)
        if reasons:
            suspicious.append({"audio_id": aid, "sample_id": audio["sample_id"],
                               "voice": audio["voice"], "condition": condition(audio),
                               "split": samples[audio["sample_id"]]["split"],
                               "path": (dataset / audio["path"]).as_posix(),
                               "duration_sec": audio["duration_sec"], "reasons": reasons,
                               "reference": rows[0]["reference_text"],
                               "whisper": rows[0]["prediction_text"], "gigaam": rows[1]["prediction_text"],
                               "word_deletions": {model: model_records[model][aid]["word_deletions"] for model in model_records}})
    counts = Counter(reason for row in suspicious for reason in row["reasons"])
    summary = {"assessment": "Preliminary evaluation on a synthetic dataset; TTS pronunciation has not been human verified and source texts have known issues.",
               "normalization": RULES, "dataset": dataset.as_posix(), "results": results.as_posix(),
               "source_sha256": sha256(source), "sample_count": len(samples), "audio_count": len(manifest),
               "sample_objects_match_original": True, "same_audio_for_both_models": True,
               "all_records_retained": True, "models": grouped, "technical": technical,
               "suspicious_audio_count": len(suspicious),
               "suspicious_clean_audio_count": sum(r["condition"] == "clean" for r in suspicious),
               "suspicious_unique_samples": len({r["sample_id"] for r in suspicious}),
               "suspicion_reason_counts": dict(counts),
               "over_25_seconds": [aid for aid, a in manifest.items() if a["duration_sec"] > 25],
               "heuristic_limits": "Flags are candidates for listening, not proven TTS faults. ASR can omit spoken digits or output numbers as words. High speaking rate and shared deletions can arise from ASR errors. Noise variants share their clean TTS source. No flags cause exclusion or reference edits."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "audio_fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n", encoding="utf-8")
    with (output / "suspected_tts.jsonl").open("x", encoding="utf-8") as stream:
        for row in suspicious:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# Предварительная оценка на синтетическом датасете", "",
             "Произношение TTS ещё не проверено человеком; в текстах есть известные проблемы.",
             "Метрики относятся ко всему синтетическому тракту TTS → ASR и не изолируют ошибки ASR.",
             "Шум — искусственный белый гауссовский, не запись реального окружения.", "",
             f"Примеры: {len(samples)}. WAV: {len(manifest)}. Обе модели обработали все WAV; хеши совпадают.",
             f"Подозрительных WAV: {len(suspicious)}; чистых: {summary['suspicious_clean_audio_count']}; уникальных sample_id: {summary['suspicious_unique_samples']}.",
             "Флаги требуют прослушивания: они не доказывают пропуск TTS. Все записи включены в оценку.", "",
             "## WER / CER / RTF", "",
             "WER/CER — проценты, RTF — время распознавания / длительность аудио; без загрузки и прогрева.",
             "Корпусные метрики: сумма ошибок / сумма эталонных единиц; не среднее отдельных WER/CER.", "",
             "| Модель | Срез | WAV | WER | CER | RTF |", "|---|---|---:|---:|---:|---:|"]
    for model, groups in grouped.items():
        for key, totals in groups.items():
            if not key.startswith("action_id="):
                lines.append(f"| {model} | {key} | {totals['records']} | {totals['wer']:.2%} | {totals['cer']:.2%} | {totals['rtf']:.3f} |")
    lines += ["", "## Флаги для прослушивания", "", "| Причина | WAV |", "|---|---:|"]
    lines.extend(f"| {reason} | {count} |" for reason, count in counts.items())
    lines += ["", "Подробности и обе гипотезы: `suspected_tts.jsonl`. Пороги: >25 символов/с; >5 слов/с;",
              "цифры в эталоне без цифр в обеих гипотезах; ≥25% удалённых слов у обеих ASR",
              "при длине каждой гипотезы <75% эталона. Последние два критерия могут отражать ASR/формат чисел.",
              "", "## Маршрут ручной проверки", "",
              "Открыть этот отчёт, `summary.json`, `suspected_tts.jsonl`, затем исходный samples.jsonl.",
              "В папке результатов открыть `whisper/predictions.jsonl`, `gigaam/predictions.jsonl` и",
              "`evaluation-*/record_metrics.jsonl`. Искать указанный audio_id и сравнивать исходный text,",
              "обе гипотезы, удаления/замены и услышанное аудио. Эталоны не исправлять по гипотезам.", ""]
    picks = []
    categories = [("ordinary", "sample_000027"), ("ordinary", "sample_000038"),
                  ("date", "sample_000039"), ("version", "sample_000157"),
                  ("technical_id", "sample_000148"), ("technical_id", "sample_000163"),
                  ("long_name", "sample_000097"), ("long_name", "sample_000103")]
    for category, sid in categories:
        for aid, a in manifest.items():
            if a["sample_id"] == sid and a["voice"] == "xenia" and not a["augmentations"]:
                picks.append((category, aid))
    # Largest absolute word error counts, plus clean examples to separate noise from TTS.
    for model, records in model_records.items():
        for scope in ("all", "clean"):
            candidates = [aid for aid in records if scope == "all" or not manifest[aid]["augmentations"]]
            for aid in sorted(candidates, key=lambda k: (records[k]["word_errors"], records[k]["cer"]), reverse=True)[:3]:
                picks.append((f"largest_word_errors_{model}_{scope}", aid))
    # Also compare the same date/ID across all voices on clean audio.
    for aid, a in manifest.items():
        if a["sample_id"] in ("sample_000039", "sample_000148") and not a["augmentations"]:
            picks.append(("compare_voices_date_or_id", aid))
    unique = {}
    for category, aid in picks:
        unique.setdefault(aid, []).append(category)
    route = []
    for aid, categories in unique.items():
        audio = manifest[aid]
        row = {"audio_id": aid, "sample_id": audio["sample_id"], "categories": categories,
               "path": (dataset / audio["path"]).as_posix(), "duration_sec": audio["duration_sec"],
               "reference": samples[audio["sample_id"]]["text"],
               **{model: records[aid]["prediction_text"] for model, records in model_records.items()}}
        route.append(row)
        lines += [f"### {aid} / {row['sample_id']} ({', '.join(categories)})", "",
                  f"WAV: `{row['path']}` ({row['duration_sec']:.3f} с).", "",
                  f"Эталон: {row['reference']}", "", f"Whisper: {row['whisper']}", "", f"GigaAM: {row['gigaam']}", ""]
    (output / "manual_review.json").write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Route paths are relative to repository root; the script resolves that root.
    repo = Path(__file__).resolve().parents[1]
    relative_repo = Path(__import__('os').path.relpath(repo, output.resolve())).as_posix()
    ps = ["param([string]$AudioId)",
          "$route = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manual_review.json') -Encoding UTF8 | ConvertFrom-Json",
          f"$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '{relative_repo}'))",
          "$route | Where-Object { -not $AudioId -or $_.audio_id -eq $AudioId } | ForEach-Object {",
          "    Write-Host $_.audio_id $_.sample_id",
          "    Write-Host $_.reference",
          "    $wav = Join-Path $repo $_.path",
          "    $player = New-Object System.Media.SoundPlayer -ArgumentList $wav",
          "    try { $player.PlaySync() } finally { $player.Dispose() }", "}"]
    (output / "listen.ps1").write_text("\n".join(ps) + "\n", encoding="utf-8")
    print(json.dumps({"audio_count": len(manifest), "suspicious": len(suspicious), "technical": technical}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("examples/zhanna-v1/samples.jsonl"))
    args = parser.parse_args()
    review(args.dataset, args.results, args.output, args.source)


if __name__ == "__main__":
    main()
