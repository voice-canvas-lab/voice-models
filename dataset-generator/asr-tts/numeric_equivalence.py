"""Conservative numeric equivalence for evaluation only; one input at a time.

No reference-dependent choices, ASR imports, or spoken-text transformations.
Scope: Russian month-name dates, explicit clock times, versions, and known IDs.
Unknown expressions remain literal and receive flags. Written zero padding is
significant; no missing fields, units, separators or zeroes are inferred.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

VERSION = "ru-numeric-eval-1.0.0"
UNITS = "ноль один два три четыре пять шесть семь восемь девять".split()
TEENS = "десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split()
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
MONTHS = dict(zip("января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), range(1, 13)))
ORDINAL = ["", "первое", "второе", "третье", "четвертое", "пятое", "шестое", "седьмое", "восьмое", "девятое", "десятое", "одиннадцатое", "двенадцатое", "тринадцатое", "четырнадцатое", "пятнадцатое", "шестнадцатое", "семнадцатое", "восемнадцатое", "девятнадцатое", "двадцатое", "двадцать первое", "двадцать второе", "двадцать третье", "двадцать четвертое", "двадцать пятое", "двадцать шестое", "двадцать седьмое", "двадцать восьмое", "двадцать девятое", "тридцатое", "тридцать первое"]
GEN_UNITS = ["", "первого", "второго", "третьего", "четвертого", "пятого", "шестого", "седьмого", "восьмого", "девятого"]
GEN_TEENS = "десятого одиннадцатого двенадцатого тринадцатого четырнадцатого пятнадцатого шестнадцатого семнадцатого восемнадцатого девятнадцатого".split()
GEN_TENS = ["", "", "двадцатого", "тридцатого", "сорокового", "пятидесятого", "шестидесятого", "семидесятого", "восьмидесятого", "девяностого"]


def cardinal(n, feminine=False):
    if n == 0:
        return "ноль"
    parts = []
    if n >= 1000:
        thousands, n = divmod(n, 1000)
        noun = "тысяча" if thousands == 1 else "тысячи" if thousands < 5 else "тысяч"
        parts.extend((cardinal(thousands, True), noun))
    if n >= 100:
        parts.append(HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    if n >= 10:
        parts.append(TEENS[n - 10])
    elif n:
        parts.append({1: "одна", 2: "две"}.get(n, UNITS[n]) if feminine else UNITS[n])
    return " ".join(parts)


def genitive_year(y):
    if y == 1900:
        return "тысяча девятисотого"
    if y == 2000:
        return "двухтысячного"
    tail = y % 100
    base = "тысяча девятьсот" if y < 2000 else "две тысячи"
    ending = (GEN_UNITS[tail] if tail < 10 else GEN_TEENS[tail - 10] if tail < 20
              else GEN_TENS[tail // 10] if tail % 10 == 0
              else TENS[tail // 10] + " " + GEN_UNITS[tail % 10])
    return base + " " + ending


NUMBERS = {tuple(cardinal(n).split()): str(n) for n in range(10000)}
NUMBERS.update({("одна",): "1", ("две",): "2"})
YEARS = {tuple(genitive_year(y).split()): str(y) for y in range(1900, 2100)}
DAYS = {tuple(s.split()): str(i) for i, s in enumerate(ORDINAL) if s}
DIGIT_WORDS = {w: str(i) for i, w in enumerate(UNITS)}
DIGIT_WORDS["нуль"] = "0"
NUMBER_WORDS = {w for phrase in NUMBERS for w in phrase} | {w for phrase in YEARS for w in phrase} | {w for phrase in DAYS for w in phrase} | {"нуль", "полтора", "полторы", "точка"}
ID_PREFIXES = {tuple(p.split()): name for p, name in (
    ("note", "note"), ("file", "file"), ("document", "document"), ("folder", "folder"),
    ("эн оу ти и", "note"), ("эф ай эл и", "file"), ("ди оу си ю эм и эн ти", "document"), ("эф оу эл ди и ар", "folder"),
    ("документ", "документ"), ("файл", "файл"), ("папка", "папка"), ("заметка", "заметка"))}
HOURS = {"час", "часа", "часов"}
MINUTES = {"минута", "минуты", "минут"}
SECONDS = {"секунда", "секунды", "секунд"}


@dataclass(frozen=True)
class Token:
    value: str
    start: int
    end: int


def lex(text):
    return [Token(unicodedata.normalize("NFC", m.group()).lower().replace("ё", "е"), m.start(), m.end())
            for m in re.finditer(r"[0-9]+|[^\W\d_]+|[^\w\s]|_", text)]


def phrase(tokens, i, table, max_words=7):
    for count in range(min(max_words, len(tokens) - i), 0, -1):
        words = tuple(t.value for t in tokens[i:i + count])
        if words in table:
            return table[words], i + count
    return None


def number(tokens, i):
    if i >= len(tokens):
        return None
    if tokens[i].value.isascii() and tokens[i].value.isdigit():
        return tokens[i].value, i + 1  # NEVER int(): leading zeroes remain.
    # Explicit leading-zero digit sequence, not a guess that e.g. 'два три' means 23.
    if tokens[i].value in ("ноль", "нуль"):
        j = i
        digits = []
        while j < len(tokens) and tokens[j].value in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[tokens[j].value])
            j += 1
        return "".join(digits), j
    return phrase(tokens, i, NUMBERS)


def canonicalize(text):
    """Single-string function. Raw text, replacements and unknown flags retained."""
    tokens = lex(text)
    values = [t.value for t in tokens]
    edits, flags = [], []
    owned = set()

    def flag(i, end, reason):
        flags.append({"start": tokens[i].start, "end": tokens[end - 1].end,
                      "text": text[tokens[i].start:tokens[end - 1].end], "reason": reason})

    def replace(i, end, kind, fields):
        canonical = " ".join(["num" + kind, *(name + value for name, value in fields), "end" + kind])
        edits.append({"start": tokens[i].start, "end": tokens[end - 1].end,
                      "before": text[tokens[i].start:tokens[end - 1].end], "after": canonical,
                      "kind": kind, "components": [{"name": k, "value": v} for k, v in fields]})
        owned.update(range(i, end))

    i = 0
    while i < len(tokens):
        # Known opaque ID: literal digits or explicit individual digit words.
        id_prefix = phrase(tokens, i, ID_PREFIXES, max_words=8)
        if id_prefix:
            prefix, j = id_prefix
            if j < len(tokens) and values[j] in ("_", "подчеркивание"):
                j += 1
                start_digits = j
                if j < len(tokens) and values[j].isascii() and values[j].isdigit():
                    digits, j = values[j], j + 1
                else:
                    chars = []
                    while j < len(tokens) and values[j] in DIGIT_WORDS:
                        chars.append(DIGIT_WORDS[values[j]])
                        j += 1
                    digits = "".join(chars)
                if digits and not (j < len(tokens) and values[j] in (".", ":", "-", "_") and j + 1 < len(tokens) and values[j + 1].isdigit()):
                    replace(i, j, "id", [("prefix", prefix), ("digits", digits)])
                    i = j
                    continue
                flag(i, max(j, start_digits), "unsupported_or_ambiguous_id")
        # Version components remain separate; only explicit version label accepted.
        if values[i] == "версия":
            component = number(tokens, i + 1)
            if component:
                components = [component[0]]
                j = component[1]
                while j < len(tokens) and values[j] in (".", "точка"):
                    next_component = number(tokens, j + 1)
                    if not next_component:
                        break  # Sentence punctuation is not a missing version component.
                    components.append(next_component[0])
                    j = next_component[1]
                if len(components) >= 2 and not (j < len(tokens) and
                        (values[j] == "точка" or values[j] in NUMBER_WORDS or values[j].isdigit())):
                    # Keep the lexical label unchanged; replace just its numeric value.
                    replace(i + 1, j, "version", [("part", v) for v in components])
                    i = j
                    continue
            flag(i, i + 1, "unsupported_or_incomplete_version")
        # Month-name date: no reordering or interpreting numeric slash/ISO forms.
        day = phrase(tokens, i, DAYS) or number(tokens, i)
        if day and day[1] < len(tokens) and values[day[1]] in MONTHS:
            month = str(MONTHS[values[day[1]]])
            year_at = day[1] + 1
            year = phrase(tokens, year_at, YEARS) or number(tokens, year_at)
            if year:
                y, j = year
                try:
                    if len(y) != 4:
                        raise ValueError("Year width")
                    date(int(y), int(month), int(day[0]))
                    if j < len(tokens) and values[j] in ("года", "год"):
                        j += 1
                    replace(i, j, "date", [("day", day[0]), ("month", month), ("year", y)])
                    i = j
                    continue
                except ValueError:
                    pass
            flag(i, year[1] if year else day[1] + 1, "unsupported_incomplete_or_invalid_date")
        # Explicit units fix component identity. Missing seconds stay missing.
        hour = number(tokens, i)
        if hour and hour[1] < len(tokens) and values[hour[1]] in HOURS:
            minute = number(tokens, hour[1] + 1)
            if minute and minute[1] < len(tokens) and values[minute[1]] in MINUTES:
                j = minute[1] + 1
                fields = [("hour", hour[0]), ("minute", minute[0])]
                second = number(tokens, j)
                if second and second[1] < len(tokens) and values[second[1]] in SECONDS:
                    fields.append(("second", second[0]))
                    j = second[1] + 1
                if int(fields[0][1]) < 24 and all(int(v) < 60 for _, v in fields[1:]):
                    replace(i, j, "time", fields)
                    i = j
                    continue
            flag(i, hour[1] + 1, "unsupported_incomplete_or_invalid_time")
        # Compact clocks need explicit 'время' context; never split a long component.
        if values[i].isascii() and values[i].isdigit() and i + 1 < len(tokens) and values[i + 1] == ":":
            j, components = i + 1, [values[i]]
            while j + 1 < len(tokens) and values[j] == ":" and values[j + 1].isascii() and values[j + 1].isdigit():
                components.append(values[j + 1])
                j += 2
            context = any(v in ("время", "времени") for v in values[max(0, i - 3):i])
            valid = (context and len(components) in (2, 3) and 1 <= len(components[0]) <= 2
                     and all(len(v) == 2 for v in components[1:])
                     and int(components[0]) < 24 and all(int(v) < 60 for v in components[1:]))
            if valid:
                replace(i, j, "time", list(zip(("hour", "minute", "second"), components)))
                i = j
                continue
            flag(i, j, "ambiguous_or_invalid_compact_time_no_repair")
        i += 1
    # Recognized number words/digits left outside supported expressions are visible.
    for i, token in enumerate(tokens):
        if i not in owned and (any(c.isdecimal() for c in token.value) or token.value in NUMBER_WORDS):
            flag(i, i + 1, "numeric_token_outside_supported_expression")
    output, pos = [], 0
    for edit in edits:
        output.extend((text[pos:edit["start"]], edit["after"]))
        pos = edit["end"]
    output.append(text[pos:])
    return {"original_text": text, "transformed_text": "".join(output),
            "rules_version": VERSION, "edits": edits, "flags": flags,
            "needs_review": bool(flags)}
