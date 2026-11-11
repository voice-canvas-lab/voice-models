"""Independent text-only speaking policy; no action/name reconstruction or ASR.

Separators can converge to spaces. This mapping is NOT character-reversible.
Metadata are carried unchanged but never supplied to transform_text().
"""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from spoken_text_v1_0_1 import number, DIGITS

RULES_VERSION = "ru-text-only-1.0.1"
NUMERIC = re.compile(r"[0-9]+(?:\.[0-9]+)*")


def transform_text(text):
    edits, flags = [], []

    def flag(start, end, reason):
        entry = {"start": start, "end": end, "original": text[start:end], "reason": reason}
        if entry not in flags:
            flags.append(entry)

    # Flag unknown pronunciation rather than inventing alphabet/meaning expansions.
    for match in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", text):
        token = match.group()
        if re.search(r"[A-Za-z]", token):
            reason = ("mixed_alphabet_word_pronunciation_unresolved" if re.search(r"[А-Яа-яЁё]", token)
                      else "mixed_alphanumeric_pronunciation_unresolved" if re.search(r"[0-9]", token)
                      else "latin_word_or_id_prefix_pronunciation_unresolved")
            flag(match.start(), match.end(), reason)
        elif re.fullmatch(r"[А-ЯЁ]{2,4}", token):
            flag(match.start(), match.end(), "possible_cyrillic_abbreviation_pronunciation_unresolved")
    for match in re.finditer(r"[0-9]+(?::[0-9]+)+", text):
        parts = match.group().split(":")
        if len(parts) not in (2, 3) or any(len(part) != 2 for part in parts[1:]):
            flag(match.start(), match.end(), "colon_numeric_components_require_pronunciation_decision")

    i = 0
    while i < len(text):
        match = NUMERIC.match(text, i)
        if match:
            token, end = match.group(), match.end()
            version = bool(re.search(r"(?<![A-Za-zА-Яа-яЁё0-9])верси[яию][\s_]+$", text[:i], re.I)) and "." in token
            components = token.split(".")
            opaque_id = bool(re.search(r"\b(?:note|file|document|folder|документ|файл|папка|заметка)_$", text[:i], re.I)) and len(token) >= 6 and token.isdigit()
            readings = [" ".join(DIGITS[int(c)] for c in part) if opaque_id else number(part) for part in components]
            if version:
                spoken, kind = " точка ".join(readings), "explicit_version"
            elif len(components) > 1:
                spoken, kind = ". ".join(readings), "numeric_components_punctuation_preserved"
                if len(components) == 2:
                    flag(i, end, "dotted_number_not_explicit_version_requires_pronunciation_decision")
            else:
                spoken = readings[0]
                kind = "technical_digit_sequence" if opaque_id else "leading_zero_digits" if len(token) > 1 and token[0] == "0" else "number_as_written"
            # Make words pronounceable without decoding an adjacent unknown letter.
            if i > 0 and (text[i - 1].isalpha() or text[i - 1] in ".:"):
                spoken = " " + spoken
            if end < len(text) and text[end].isalpha():
                spoken += " "
            edits.append({"start": i, "end": end, "original": token, "spoken": spoken,
                          "reason": kind, "numeric_components": components, "component_readings": readings})
            i = end
            continue
        if text[i] == "_":
            edits.append({"start": i, "end": i + 1, "original": "_", "spoken": " ", "reason": "underscore_as_word_boundary"})
        elif text[i] == "-":
            if i + 1 < len(text) and text[i + 1].isdigit() and (i == 0 or text[i - 1].isspace()):
                # A minus is not safely interchangeable with a filename hyphen.
                flag(i, i + 1, "possible_signed_number_preserved_requires_decision")
            elif i > 0 and i + 1 < len(text) and not text[i - 1].isspace() and not text[i + 1].isspace():
                edits.append({"start": i, "end": i + 1, "original": "-", "spoken": " ", "reason": "hyphen_as_word_boundary"})
        i += 1
    parts, pos = [], 0
    for edit in edits:
        parts.extend((text[pos:edit["start"]], edit["spoken"]))
        pos = edit["end"]
    parts.append(text[pos:])
    return {"spoken_text": "".join(parts), "edits": edits, "pronunciation_flags": flags,
            "needs_pronunciation_review": bool(flags), "tts_ready": not flags,
            "policy": "text_only_no_label_repair_not_character_reversible"}


def derive(source, output, previous_unresolved=None):
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    mappings, derived, ready, review = [], [], [], []
    for row in rows:
        result = transform_text(row["text"])  # The only input to the transformation.
        item = copy.deepcopy(row)
        item["text"] = result["spoken_text"]
        derived.append(item)
        record = {"sample_id": row["sample_id"], "original_text": row["text"], "rules_version": RULES_VERSION, **result}
        mappings.append(record)
        (review if result["needs_pronunciation_review"] else ready).append(record if result["needs_pronunciation_review"] else item)
    reconsidered = []
    if previous_unresolved:
        previous = [json.loads(line) for line in previous_unresolved.read_text(encoding="utf-8").splitlines()]
        index = {m["sample_id"]: m for m in mappings}
        for old in previous:
            current = index[old["sample_id"]]
            reconsidered.append({"sample_id": old["sample_id"], "original_text": current["original_text"],
                                 "spoken_text": current["spoken_text"], "previous_reasons": old["transformation_review_reasons"],
                                 "previous_evidence": old["review_evidence"],
                                 "action_label_issues_block_audio": False,
                                 "current_status": "pronunciation_review_required" if current["needs_pronunciation_review"] else "ready_under_text_only_policy",
                                 "current_pronunciation_flags": current["pronunciation_flags"]})
        assert len(reconsidered) == 63
    for filename, records in (("samples.jsonl", derived), ("mapping.jsonl", mappings),
                              ("tts_ready_samples.jsonl", ready), ("pronunciation_review.jsonl", review),
                              ("reconsidered_63.jsonl", reconsidered)):
        with (output / filename).open("x", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"rules_version": RULES_VERSION, "source_sha256": hashlib.sha256(raw).hexdigest(),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "number_helper_sha256": hashlib.sha256(Path(__file__).with_name("spoken_text_v1_0_1.py").read_bytes()).hexdigest(),
               "samples": len(rows), "changed_texts": sum(a["text"] != b["text"] for a, b in zip(rows, derived)),
               "tts_ready": len(ready), "pronunciation_review": len(review),
               "previous_63_released": sum(r["current_status"] == "ready_under_text_only_policy" for r in reconsidered),
               "previous_63_still_pronunciation_review": sum(r["current_status"] == "pronunciation_review_required" for r in reconsidered),
               "reason_counts": dict(Counter(e["reason"] for m in mappings for e in m["edits"])),
               "pronunciation_issue_counts": {},
               "reads_args_context_action_id": False, "label_issues_block_audio": False,
               "character_reversible": False, "candidate_text_is_tts_input_and_asr_reference": True}
    summary["pronunciation_issue_counts"] = dict(Counter(reason for m in mappings for reason in {f["reason"] for f in m["pronunciation_flags"]}))
    (output / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Произносимая версия только из text", "", "Правила: " + RULES_VERSION,
             "Все исходные строки сохранены. Разметка действий не является основанием исключения.",
             "Это правило озвучивания, не восстановление точного имени и не посимвольно обратимое кодирование.", ""]
    for m in mappings:
        lines += ["## " + m["sample_id"], "", "Было: " + m["original_text"], "", "Стало: " + m["spoken_text"], "",
                  "Вопросы произношения: " + json.dumps(m["pronunciation_flags"], ensure_ascii=False), ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    previous_lines = ["# Пересмотр 63 прежних исключений", "", "Проблемы args/context/action_id не блокируют аудиопрогон и не исправляются.",
                      "Флаги ниже относятся только к произношению исходного текста.", "",
                      "| sample_id | Исходный text | Прежняя причина | Новый статус | Вопрос произношения |", "|---|---|---|---|---|"]
    for m in reconsidered:
        values = [m["sample_id"], m["original_text"], '; '.join(m["previous_reasons"]), m["current_status"],
                  json.dumps(m["current_pronunciation_flags"], ensure_ascii=False)]
        previous_lines.append('| ' + ' | '.join(v.replace('|', '\\|') for v in values) + ' |')
    (output / "RECONSIDERED_63.md").write_text('\n'.join(previous_lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-unresolved", type=Path)
    args = parser.parse_args()
    derive(args.source, args.output, args.previous_unresolved)


if __name__ == "__main__":
    main()
