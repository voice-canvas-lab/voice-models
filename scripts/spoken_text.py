"""Contract-bound exact filenames for TTS/reference preparation, not ASR hints."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from spoken_text_v2_0_0 import analyze as analyze_outside

RULES_VERSION = "ru-spoken-3.1.0"
# Explicit semantic roles for the frozen action catalog. A query is NOT a filename.
NAME_ROLES = {
    "create": {"name": "object", "parent_name": "folder"},
    "open": {"name": "object"}, "delete": {"name": "object"},
    "rename": {"old_name": "object", "new_name": "object"},
    "move": {"name": "object", "destination_name": "folder"},
    "copy": {"name": "object", "new_name": "object", "destination_name": "folder"},
    "save": {"name": "object"}, "search": {},
}
RUSSIAN = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["а", "бэ", "вэ", "гэ", "дэ", "е", "ё", "жэ", "зэ", "и", "и краткая", "ка", "эл", "эм", "эн", "о", "пэ", "эр", "эс", "тэ", "у", "эф", "ха", "цэ", "че", "ша", "ща", "твёрдый знак", "ы", "мягкий знак", "э", "ю", "я"]))
LATIN = dict(zip("abcdefghijklmnopqrstuvwxyz",
    ["эй", "би", "си", "ди", "и", "эф", "джи", "эйч", "ай", "джей", "кей", "эл", "эм", "эн", "оу", "пи", "кью", "ар", "эс", "ти", "ю", "ви", "дабл ю", "экс", "уай", "зед"]))
DIGITS = "ноль один два три четыре пять шесть семь восемь девять".split()
SIGNS = {"-": "дефис", "_": "подчёркивание", ".": "точка", " ": "пробел",
         "(": "левая круглая скобка", ")": "правая круглая скобка",
         "[": "левая квадратная скобка", "]": "правая квадратная скобка",
         ",": "запятая", "+": "плюс"}


def speak_filename(name):
    """Ordinary words as words, extension/technical runs as letters, digits singly.

    Case/alphabet remain in metadata, never announced. Unknown Latin words and
    mixed alphabets require a manual decision instead of guessing a reading.
    """
    if not name or name != name.strip() or "  " in name:
        raise ValueError("Ambiguous filename whitespace; manual decision required")
    technical = {"api", "id", "uuid", "xml", "json", "csv", "pdf", "txt", "md",
                 "doc", "docx", "xls", "xlsx", "sql", "asr", "tts", "tar", "gz", "zip"}
    last_dot = name.rfind(".")
    extension_start = last_dot + 1 if 0 < last_dot < len(name) - 1 else None
    segments = []
    for match in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+|.", name, flags=re.DOTALL):
        token = match.group()
        letters = [c for c in token if c.isalpha()]
        if any(c.lower() not in RUSSIAN and c.lower() not in LATIN for c in letters):
            raise ValueError("Unsupported filename alphabet")
        if letters and any(c.lower() in RUSSIAN for c in letters) and any(c.lower() in LATIN for c in letters):
            raise ValueError("Mixed alphabets in one filename component; manual decision required")
        extension = extension_start is not None and match.start() >= extension_start
        is_technical = extension or (letters and any(c.isdigit() for c in token)) or token.lower() in technical
        if token == " ":
            spoken, kind = " ", "word_boundary"
        elif token.isascii() and token.isdigit():
            spoken, kind = " ".join(DIGITS[int(c)] for c in token), "digits_preserve_zeros"
        elif letters and is_technical:
            spoken = " ".join(DIGITS[int(c)] if c in "0123456789" else
                              RUSSIAN[c.lower()] if c.lower() in RUSSIAN else LATIN[c.lower()]
                              for c in token)
            kind = "extension_letters" if extension else "technical_sequence_letters"
        elif letters and all(c.lower() in RUSSIAN for c in letters):
            spoken, kind = token, "ordinary_word"
        elif letters:
            raise ValueError("Unknown Latin filename word; manual pronunciation decision required")
        elif token in SIGNS:
            spoken, kind = SIGNS[token], "significant_filename_separator"
        else:
            raise ValueError(f"Unsupported exact filename character at {match.start()}: U+{ord(token):04X}")
        segments.append({"start": match.start(), "end": match.end(), "original": token,
                         "spoken": spoken, "kind": kind})
    return " ".join(s["spoken"] for s in segments if s["kind"] != "word_boundary"), segments


def reliable_boundary(text, start, end):
    # A literal args value may still be only a substring of a larger filename.
    for pos, direction in ((start - 1, -1), (end, 1)):
        if not 0 <= pos < len(text):
            continue
        char = text[pos]
        if char.isalnum() or char == "_":
            return False
        if char in ".-" and 0 <= pos + direction < len(text) and text[pos + direction].isalnum():
            return False
    return True


def occurrences(text, value):
    result, start = [], 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            return result
        result.append((pos, pos + len(value)))
        start = pos + 1


def analyze_sample(sample, actions):
    text, args = sample["text"], sample["args"]
    action_id = sample["action_id"]
    flags, bindings = [], []

    def flag(reason, **evidence):
        flags.append({"reason": reason, **evidence})

    action = actions.get(action_id)
    if action is None or action_id not in NAME_ROLES:
        flag("unsupported_action_contract", action_id=action_id)
    else:
        schema = action["parameters"]
        properties = schema["properties"]
        if not set(schema.get("required", [])) <= args.keys() or (schema.get("additionalProperties") is False and set(args) - properties.keys()):
            flag("args_do_not_match_action_schema")
        for key, value in args.items():
            prop = properties.get(key, {})
            if prop.get("type") == "string" and (not isinstance(value, str) or not value.strip()):
                flag("invalid_string_arg", argument=key)
            if "enum" in prop and value not in prop["enum"]:
                flag("arg_not_in_schema_enum", argument=key)
        for key, role in NAME_ROLES[action_id].items():
            if key not in args:
                continue
            value = args[key]
            if properties.get(key, {}).get("type") != "string" or not isinstance(value, str) or not value:
                flag("unsupported_name_parameter_contract", argument=key)
                continue
            matches = occurrences(text, value)
            if len(matches) != 1:
                flag("name_absent_or_repeated_in_text", argument=key, args_name=value, occurrences=len(matches))
                continue
            start, end = matches[0]
            if not reliable_boundary(text, start, end):
                flag("unreliable_name_boundaries", argument=key, args_name=value, start=start, end=end)
                continue
            object_type = args.get("object_type") if role == "object" else role
            if object_type is None:
                flag("name_object_type_not_known_from_args", argument=key)
                continue
            binding = {"argument": key, "role": role, "object_type": object_type,
                       "start": start, "end": end, "original_name": value,
                       "contract_action_id": action_id, "contract_property": properties[key]}
            if object_type == "file":
                try:
                    binding["spoken_name"], binding["segments"] = speak_filename(value)
                    binding["characters"] = [{"index": i, "character": c,
                                              "case": "upper" if c.isupper() else "lower" if c.islower() else "none"}
                                             for i, c in enumerate(value)]
                except ValueError as error:
                    flag("filename_pronunciation_requires_review", argument=key, detail=str(error))
                    continue
            bindings.append(binding)
    ordered = sorted(bindings, key=lambda b: (b["start"], b["end"]))
    for previous, current in zip(ordered, ordered[1:]):
        if previous["end"] > current["start"]:
            flag("overlapping_or_shared_name_roles", arguments=[previous["argument"], current["argument"]])
    exact_files = [b for b in ordered if b["object_type"] == "file"]
    parts, edits, pos = [], [], 0
    if not flags:
        for binding in exact_files:
            outside = analyze_outside(text[pos:binding["start"]])
            for reason in outside["review_reasons"]:
                flag("outside_name_requires_review", detail=reason)
            parts.extend((outside["spoken_text"], binding["spoken_name"]))
            edits.extend({**e, "start": pos + e["start"], "end": pos + e["end"]} for e in outside["edits"])
            edits.append({"start": binding["start"], "end": binding["end"], "original": binding["original_name"],
                          "spoken": binding["spoken_name"], "kind": "contract_bound_exact_filename", "argument": binding["argument"]})
            pos = binding["end"]
        outside = analyze_outside(text[pos:])
        for reason in outside["review_reasons"]:
            flag("outside_name_requires_review", detail=reason)
        parts.append(outside["spoken_text"])
        edits.extend({**e, "start": pos + e["start"], "end": pos + e["end"]} for e in outside["edits"])
    eligible = not flags
    return {"spoken_text": "".join(parts) if eligible else text,
            "auto_tts_eligible": eligible, "review_reasons": sorted({f["reason"] for f in flags}),
            "review_evidence": flags, "edits": edits if eligible else [],
            "name_bindings": bindings, "filename_mappings_applied": eligible}


def derive(source, output, actions_path):
    data = source.read_bytes()
    rows = [json.loads(l) for l in data.decode("utf-8-sig").splitlines()]
    actions = {a["action_id"]: a for a in map(json.loads, actions_path.read_text(encoding="utf-8").splitlines())}
    output.mkdir(parents=True, exist_ok=False)
    all_rows, safe, mapping, names, manual = [], [], [], [], []
    for row in rows:
        result = analyze_sample(row, actions)
        derived = copy.deepcopy(row)
        derived["text"] = result["spoken_text"]
        all_rows.append(derived)
        record = {"sample_id": row["sample_id"], "original_text": row["text"], "rules_version": RULES_VERSION, **result}
        mapping.append(record)
        if result["auto_tts_eligible"]:
            safe.append(derived)
        else:
            manual.append(record)
        for binding in result["name_bindings"]:
            if binding["object_type"] == "file":
                names.append({"sample_id": row["sample_id"], "rules_version": RULES_VERSION,
                              "applied": result["filename_mappings_applied"], **binding})
    for filename, records in (("samples.jsonl", all_rows), ("safe_samples.jsonl", safe),
                              ("mapping.jsonl", mapping), ("filename_mapping.jsonl", names), ("manual_review.jsonl", manual)):
        with (output / filename).open("x", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"rules_version": RULES_VERSION, "source_sha256": hashlib.sha256(data).hexdigest(),
               "actions_sha256": hashlib.sha256(actions_path.read_bytes()).hexdigest(),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "outside_v2_sha256": hashlib.sha256(Path(__file__).with_name("spoken_text_v2_0_0.py").read_bytes()).hexdigest(),
               "samples": len(rows), "auto_tts_eligible": len(safe), "manual_review": len(manual),
               "exact_filename_bindings": len(names), "applied_filename_bindings": sum(n["applied"] for n in names),
               "review_reason_counts": dict(Counter(reason for row in manual for reason in row["review_reasons"])),
               "policy": "Contract-bound exact names with the same v3 boundary checks. Ordinary Cyrillic words are read as words; extension and alphanumeric/known technical sequences are spelled, digits individually, meaningful signs retained. No spoken alphabet or case labels. Original case/characters retained in metadata. Unknown Latin words, mixed alphabets and ambiguous whitespace require manual decision. Dates/version semantics never override file spelling. Outside names use v2. Derived text is both TTS input and ASR reference; metadata is never an ASR prompt."}
    (output / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# " + RULES_VERSION, "", "Текстовая подготовка; полный синтез/ASR не выполнялись.",
             "Все строки сохранены. Только safe_samples.jsonl допускается к TTS после выбора объёма.", ""]
    for record in mapping:
        lines += ["## " + record["sample_id"], "", "Было: " + record["original_text"], "", "Стало: " + record["spoken_text"],
                  "", "Авто-TTS: " + str(record["auto_tts_eligible"]), "", "Флаги: " + ", ".join(record["review_reasons"]), ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "actions"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    derive(args.source, args.output, args.actions)


if __name__ == "__main__":
    main()
