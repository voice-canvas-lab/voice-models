"""Contract-bound exact filenames for TTS/reference preparation, not ASR hints."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from spoken_text_v2_0_0 import analyze as analyze_outside

RULES_VERSION = "ru-spoken-3.0.0"
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
    """Every character is represented; unsupported characters block the sample.

    Alphabet and uppercase are audible. Unmarked letters are lowercase.
    Extension/date/version detection never overrides exact-name spelling.
    """
    chars = []
    for i, char in enumerate(name):
        lower = char.lower()
        if lower in RUSSIAN:
            spoken = ("заглавная " if char.isupper() else "") + "русская " + RUSSIAN[lower]
        elif lower in LATIN:
            spoken = ("заглавная " if char.isupper() else "") + "латинская " + LATIN[lower]
        elif char in "0123456789":
            spoken = DIGITS[int(char)]
        elif char in SIGNS:
            spoken = SIGNS[char]
        else:
            raise ValueError(f"Unsupported exact filename character at {i}: U+{ord(char):04X}")
        chars.append({"index": i, "character": char, "spoken": spoken})
    return " ".join(c["spoken"] for c in chars), chars


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
                    binding["spoken_name"], binding["characters"] = speak_filename(value)
                except ValueError as error:
                    flag("unsupported_filename_character", argument=key, detail=str(error))
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
               "policy": "Exact case-sensitive args name, explicit action-schema role, single occurrence, reliable boundaries, no overlapping roles. File characters retain alphabet/case, spaces, signs and every digit. Dates/version priority is overridden inside exact filenames. Outside spans use v2. Uncertain whole rows remain unchanged. Derived text is both TTS input and new ASR reference; args are preprocessing metadata only, never ASR prompts."}
    (output / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# ru-spoken-3.0.0", "", "Текстовая подготовка; полный синтез/ASR не выполнялись.",
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
