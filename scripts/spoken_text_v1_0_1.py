"""Deterministic, text-only Russian spoken form; no model or label access."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

RULES_VERSION = "ru-spoken-1.0.1"
DIGITS = "ноль один два три четыре пять шесть семь восемь девять".split()
TEENS = "десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split()
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
ORDINAL = ["", "первое", "второе", "третье", "четвёртое", "пятое", "шестое", "седьмое", "восьмое", "девятое", "десятое", "одиннадцатое", "двенадцатое", "тринадцатое", "четырнадцатое", "пятнадцатое", "шестнадцатое", "семнадцатое", "восемнадцатое", "девятнадцатое", "двадцатое", "двадцать первое", "двадцать второе", "двадцать третье", "двадцать четвёртое", "двадцать пятое", "двадцать шестое", "двадцать седьмое", "двадцать восьмое", "двадцать девятое", "тридцатое", "тридцать первое"]
YEAR_UNITS = ["", "первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого", "восьмого", "девятого"]
YEAR_TEENS = "десятого одиннадцатого двенадцатого тринадцатого четырнадцатого пятнадцатого шестнадцатого семнадцатого восемнадцатого девятнадцатого".split()
YEAR_TENS = {20: "двадцатого", 30: "тридцатого", 40: "сорокового", 50: "пятидесятого", 60: "шестидесятого", 70: "семидесятого", 80: "восьмидесятого", 90: "девяностого"}
SEPARATORS = {"_": "подчёркивание", ".": "точка", "-": "дефис", ":": "двоеточие"}
LATIN_NAMES = {"note": "эн оу ти и", "file": "эф ай эл и", "document": "ди оу си ю эм и эн ти", "folder": "эф оу эл ди и ар"}
ID = re.compile(r"(?<!\w)(?:note|file|document|folder|заметка|файл|документ|папка)_[0-9]+(?!\w|[.:\-][0-9])", re.I)
NUMBER_CHAIN = re.compile(r"[0-9]+(?:[._:\-][0-9]+)*")


def inflect(n, forms):
    if 11 <= n % 100 <= 14:
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1] if 2 <= n % 10 <= 4 else forms[2]


def cardinal(n, feminine=False):
    if not 0 <= n < 10**12:
        raise ValueError("Cardinal outside supported range")
    if n == 0:
        return DIGITS[0]
    result = []
    for value, forms, gender in ((10**9, ("миллиард", "миллиарда", "миллиардов"), False),
                                 (10**6, ("миллион", "миллиона", "миллионов"), False),
                                 (1000, ("тысяча", "тысячи", "тысяч"), True)):
        count, n = divmod(n, value)
        if count:
            result.extend((cardinal(count, gender), inflect(count, forms)))
    if n >= 100:
        result.append(HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        result.append(TENS[n // 10])
        n %= 10
    if n >= 10:
        result.append(TEENS[n - 10])
    elif n:
        result.append({1: "одна", 2: "две"}.get(n, DIGITS[n]) if feminine else DIGITS[n])
    return " ".join(result)


def number(token):
    # Retain every leading zero; large/opaque values are read digit by digit.
    if len(token) > 1 and token[0] == "0" or len(token) > 12:
        return " ".join(DIGITS[int(c)] for c in token)
    return cardinal(int(token))


def year_genitive(year):
    if year == 1900:
        return "тысяча девятисотого"
    if year == 2000:
        return "двухтысячного"
    if not 1900 < year < 2100:
        raise ValueError("Natural date year outside 1900..2099")
    base = "тысяча девятьсот" if year < 2000 else "две тысячи"
    tail = year % 100
    if tail < 10:
        ending = YEAR_UNITS[tail]
    elif tail < 20:
        ending = YEAR_TEENS[tail - 10]
    elif tail % 10 == 0:
        ending = YEAR_TENS[tail]
    else:
        ending = TENS[tail // 10] + " " + YEAR_UNITS[tail % 10]
    return base + " " + ending


def date_words(token):
    y, m, d = map(int, token.split("-"))
    date(y, m, d)  # Calendar validation, including leap years.
    return f"{ORDINAL[d]} {MONTHS[m - 1]} {year_genitive(y)} года"


def time_words(token):
    parts = list(map(int, re.split(r"[.:]", token)))
    if len(parts) not in (2, 3) or not 0 <= parts[0] < 24 or any(not 0 <= p < 60 for p in parts[1:]):
        raise ValueError("Invalid clock time")
    forms = [("час", "часа", "часов"), ("минута", "минуты", "минут"), ("секунда", "секунды", "секунд")]
    return " ".join(cardinal(n, feminine=i > 0) + " " + inflect(n, forms[i]) for i, n in enumerate(parts))


def literal_chain(token):
    return " ".join(SEPARATORS[p] if p in SEPARATORS else number(p)
                    for p in re.split(r"([._:\-])", token))


def transform(text):
    """Return spoken text, ordered reversible span edits, and review flags.

    Only text is supplied: args/context never fill missing values or fix errors.
    Unknown numeric syntax is read literally and explicitly marked for review.
    """
    edits, flags = [], []
    i = 0
    def add(start, end, spoken, kind, warnings=()):
        edits.append({"start": start, "end": end, "original": text[start:end],
                      "spoken": spoken, "kind": kind, "review_reasons": list(warnings)})
        flags.extend(warnings)
    while i < len(text):
        match = ID.match(text, i)
        if match:
            prefix, digits = match.group().rsplit("_", 1)
            spoken_prefix = LATIN_NAMES.get(prefix.lower(), prefix)
            add(i, match.end(), spoken_prefix + " подчёркивание " + " ".join(DIGITS[int(c)] for c in digits),
                "technical_id", ["latin_id_letter_names_require_listening"] if prefix.lower() in LATIN_NAMES else [])
            i = match.end()
            continue
        match = NUMBER_CHAIN.match(text, i)
        if match:
            token, end = match.group(), match.end()
            prefix = text[:i]
            warnings = []
            # Calendar date and date_time only in exact ISO date + dotted/colon time syntax.
            dt = re.fullmatch(r"([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2}([.:])[0-9]{2}\3[0-9]{2})", token)
            iso = re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", token)
            version = re.search(r"верси[яию][ _]+$", prefix, re.I) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", token)
            clock = re.search(r"(?:время|времени)[ :_]+$", prefix, re.I) and re.fullmatch(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?", token)
            try:
                if dt:
                    spoken = date_words(dt[1]) + " " + time_words(dt[2])
                    kind = "iso_datetime"
                elif iso:
                    spoken, kind = date_words(token), "iso_date"
                elif version:
                    spoken, kind = literal_chain(token), "explicit_version"
                elif clock:
                    spoken, kind = time_words(token), "explicit_clock_time"
                elif token.isdigit():
                    # Opaque numeric suffixes and alphanumeric designators retain digits literally.
                    if (i > 0 and text[i - 1] == ".") or (end < len(text) and text[end].isalpha()) or (i > 0 and text[i - 1].isalpha()):
                        spoken, kind = number(token), "literal_numeric_component"
                        warnings.append("ambiguous_name_suffix_or_alphanumeric")
                    else:
                        spoken, kind = number(token), "cardinal_number"
                else:
                    spoken, kind = literal_chain(token), "literal_numeric_chain"
                    warnings.append("ambiguous_numeric_chain_no_semantic_guess")
                    if re.match(r"[0-9]{4,}-[0-9]{2}-[0-9]{2}", token):
                        warnings.append("noncanonical_or_invalid_datetime_like_sequence")
            except ValueError:
                spoken, kind = literal_chain(token), "literal_invalid_date_or_time"
                warnings.append("invalid_or_unsupported_date_time_preserved_literally")
            add(i, end, spoken, kind, warnings)
            i = end
            continue
        if text[i] == "_":
            add(i, i + 1, " подчёркивание ", "literal_name_separator")
        elif text[i] == "." and i + 1 < len(text) and text[i + 1].isdigit():
            add(i, i + 1, " точка ", "literal_name_dot", ["ambiguous_name_numeric_suffix"])
        elif text[i] in "Dd" and i > 0 and text[i - 1].isdigit():
            add(i, i + 1, " ди", "literal_latin_designator", ["mixed_alphanumeric_requires_review"])
        i += 1
    spoken, pos = [], 0
    for edit in edits:
        spoken.extend((text[pos:edit["start"]], edit["spoken"]))
        pos = edit["end"]
    spoken.append(text[pos:])
    return "".join(spoken), edits, sorted(set(flags))


def derive(source, output):
    data = source.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    mappings, derived = [], []
    for row in rows:
        spoken, edits, flags = transform(row["text"])
        changed = copy.deepcopy(row)
        changed["text"] = spoken
        derived.append(changed)
        mappings.append({"sample_id": row["sample_id"], "original_text": row["text"],
                         "spoken_text": spoken, "rules_version": RULES_VERSION,
                         "changed": spoken != row["text"], "needs_review": bool(flags),
                         "review_reasons": flags, "edits": edits})
    for name, records in (("samples.jsonl", derived), ("mapping.jsonl", mappings)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"rules_version": RULES_VERSION, "source": source.as_posix(),
               "source_sha256": hashlib.sha256(data).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "samples": len(rows), "changed": sum(m["changed"] for m in mappings),
               "needs_review": sum(m["needs_review"] for m in mappings),
               "rule_counts": dict(Counter(e["kind"] for m in mappings for e in m["edits"])),
               "review_reason_counts": dict(Counter(f for m in mappings for f in m["review_reasons"])),
               "reference_policy": "Only derived text/spoken_text is the reference of a new experiment; other sample fields are unchanged.",
               "policy": "Text only; no LLM, no labels/context, no repairs. ISO dates validated; explicit versions/time recognized. Ambiguity uses literal reading with separator words and review flags. Technical IDs retain every digit and spell known Latin prefixes. Non-numeric text is retained exactly outside logged edits."}
    (output / "rules.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Производная произносимая версия Жанны", "", f"Правила: `{RULES_VERSION}`.",
             f"Всего: {len(rows)}; изменено: {summary['changed']}; требует проверки: {summary['needs_review']}.",
             "Исходный снимок не изменён. Только text заменён; остальные поля сохранены.",
             "Эталон нового эксперимента — spoken_text. Это отдельная версия датасета.",
             "Полный набор не синтезирован. Неоднозначные записи прочитаны буквально и отмечены.", "",
             "ISO-даты читаются по календарю (1900–2099); время — только после явного слова время или в каноническом timestamp.",
             "Версии распознаются после слова версия (включая подчёркивание). Обычные числа — количественные, без догадки о падеже/годе.",
             "Неоднозначные цифровые цепочки читаются с названиями исходных разделителей; опечатки не исправляются.",
             "ID читаются по цифрам с ведущими нулями; латинские префиксы — по именам букв, кириллические — как исходное слово.",
             "Все изменения записаны как непересекающиеся исходные интервалы start/end в mapping.jsonl.",
             "Текст вне интервалов сохранён буквально. Никакие значения из args/context не подставляются.", ""]
    for m in mappings:
        lines += [f"## {m['sample_id']}" + (" — проверить" if m["needs_review"] else ""), "",
                  f"Было: {m['original_text']}", "", f"Стало: {m['spoken_text']}", "",
                  "Правила: " + ", ".join(e["kind"] for e in m["edits"]), "",
                  "Замечания: " + ", ".join(m["review_reasons"]), ""]
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
