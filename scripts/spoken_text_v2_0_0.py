"""Safe Russian spoken form v2: punctuation stays punctuation, ambiguity blocks TTS."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from spoken_text_v1_0_1 import date_words, time_words, number, ID, NUMBER_CHAIN

RULES_VERSION = "ru-spoken-2.0.0"


def analyze(text):
    edits, decisions = [], []
    # An underscore within a name is not automatically a pause or a dictation cue.
    opaque = list(re.finditer(r'[^\s,;!?«»"()]+_[^\s,;!?«»"()]+', text))
    for match in opaque:
        token = match.group().rstrip(".")
        stamp = re.fullmatch(r"([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}([.:])[0-9]{2}\3[0-9]{2})", token)
        if stamp:
            try:
                date_words(stamp[1])
                time_words(stamp[2])
                continue  # A complete standalone timestamp, not a named object.
            except ValueError:
                pass
        technical = ID.fullmatch(token)
        decisions.append({"start": match.start(), "end": match.start() + len(token),
                          "original": token, "reason": "technical_id_manual_decision_required" if technical
                          else "name_separator_manual_decision_required",
                          "action": "preserve_original_and_queue_separate_technical_test"})

    def blocked(i):
        return any(d["start"] <= i < d["end"] for d in decisions)

    def review(start, end, reason):
        decisions.append({"start": start, "end": end, "original": text[start:end],
                          "reason": reason, "action": "preserve_original_and_queue_separate_technical_test"})

    i = 0
    while i < len(text):
        if blocked(i):
            i += 1
            continue
        match = NUMBER_CHAIN.match(text, i)
        if not match:
            i += 1
            continue
        token, end = match.group(), match.end()
        prefix = text[:i]
        dt = re.fullmatch(r"([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}([.:])[0-9]{2}\3[0-9]{2})", token)
        iso = re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", token)
        version = re.search(r"\bверси[яию]\s+$", prefix, re.I) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", token)
        clock = re.search(r"\b(?:время|времени)[\s:—–-]*$", prefix, re.I) and re.fullmatch(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?", token)
        attached = (i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_.-")) or (end < len(text) and text[end].isalpha())
        try:
            if attached:
                review(i, end, "ambiguous_name_suffix_or_alphanumeric_manual_decision_required")
            elif version:
                spoken = " точка ".join(number(component) for component in token.split("."))
                edits.append({"start": i, "end": end, "original": token, "spoken": spoken, "kind": "explicit_version"})
            elif dt:
                spoken = date_words(dt[1]) + ", " + time_words(dt[2])
                edits.append({"start": i, "end": end, "original": token, "spoken": spoken, "kind": "structured_datetime"})
            elif iso:
                spoken = date_words(token)
                edits.append({"start": i, "end": end, "original": token, "spoken": spoken, "kind": "structured_date"})
            elif clock:
                spoken = time_words(token)
                edits.append({"start": i, "end": end, "original": token, "spoken": spoken, "kind": "structured_time"})
            elif token.isdigit() and not (len(token) > 1 and token.startswith("0")) and len(token) <= 12:
                edits.append({"start": i, "end": end, "original": token, "spoken": number(token), "kind": "cardinal_number"})
            else:
                review(i, end, "ambiguous_numeric_expression_manual_decision_required")
        except ValueError:
            review(i, end, "invalid_or_unsupported_date_time_manual_decision_required")
        i = end
    flags = sorted({d["reason"] for d in decisions})
    # Atomic policy: do not make a partly normalized command look ready for TTS.
    if flags:
        return {"spoken_text": text, "edits": [], "review_reasons": flags,
                "decisions": decisions, "proposed_safe_edits_not_applied": edits, "auto_tts_eligible": False}
    result, pos = [], 0
    for edit in edits:
        result.extend((text[pos:edit["start"]], edit["spoken"]))
        pos = edit["end"]
    result.append(text[pos:])
    return {"spoken_text": "".join(result), "edits": edits, "review_reasons": [],
            "decisions": [], "proposed_safe_edits_not_applied": [], "auto_tts_eligible": True}


def transform(text):
    result = analyze(text)
    return result["spoken_text"], result["edits"], result["review_reasons"]


def derive(source, output):
    data = source.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    mapping, all_rows, safe, manual = [], [], [], []
    for row in rows:
        result = analyze(row["text"])
        derived = copy.deepcopy(row)
        derived["text"] = result["spoken_text"]
        all_rows.append(derived)
        mapping.append({"sample_id": row["sample_id"], "original_text": row["text"],
                        "rules_version": RULES_VERSION, "changed": derived["text"] != row["text"],
                        "needs_review": not result["auto_tts_eligible"], **result})
        if result["auto_tts_eligible"]:
            safe.append(derived)
        else:
            manual.append({"sample_id": row["sample_id"], "original_text": row["text"],
                           "rules_version": RULES_VERSION, "decisions": result["decisions"],
                           "status": "manual_decision_required_before_separate_technical_test",
                           "automatic_dictation": False, "audio_generated": False})
    for filename, records in (("samples.jsonl", all_rows), ("safe_samples.jsonl", safe),
                              ("mapping.jsonl", mapping), ("technical_test_queue.jsonl", manual)):
        with (output / filename).open("x", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"rules_version": RULES_VERSION, "source": str(source),
               "source_sha256": hashlib.sha256(data).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "helper_v1_sha256": hashlib.sha256(Path(__file__).with_name("spoken_text_v1_0_1.py").read_bytes()).hexdigest(),
               "samples": len(rows), "safe_samples": len(safe), "manual_decision_required": len(manual),
               "changed": sum(r["changed"] for r in mapping),
               "applied_rule_counts": dict(Counter(e["kind"] for r in mapping for e in r["edits"])),
               "review_reason_counts": dict(Counter(reason for r in mapping for reason in r["review_reasons"])),
               "policy": "Punctuation retained for pauses. Only validated structured dates/time, explicit versions, ordinary numbers. No generic separator naming or automatic ID spelling. Ambiguous whole commands preserved verbatim and excluded from auto TTS input; separate technical queue requires manual decisions. No labels/context are consulted."}
    (output / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# ru-spoken-2.0.0", "", "samples.jsonl сохраняет все строки; для обычного TTS допускается ТОЛЬКО safe_samples.jsonl.",
             "Неоднозначные строки сохранены буквально и вынесены в technical_test_queue.jsonl. Они не синтезировались.",
             "Пунктуация остаётся пунктуацией; точка произносится только внутри явной версии.", ""]
    for r in mapping:
        lines += ["## " + r["sample_id"], "", "Было: " + r["original_text"], "", "Стало: " + r["spoken_text"],
                  "", "Авто-TTS: " + str(r["auto_tts_eligible"]), "", "Флаги: " + ", ".join(r["review_reasons"]), ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.output)


if __name__ == "__main__":
    main()
