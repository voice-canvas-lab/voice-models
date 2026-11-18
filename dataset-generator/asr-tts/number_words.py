"""Russian number words used by the text-only TTS policy."""

DIGITS = "ноль один два три четыре пять шесть семь восемь девять".split()

TEENS = "десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split()

TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]

HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

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
