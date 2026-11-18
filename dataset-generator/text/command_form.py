"""Проверка формы голосовой команды."""

import re


QUOTED_TITLE = re.compile(r'«[^»]*»|“[^”]*”|"[^"]*"')
CONDITIONAL_CLAUSE = re.compile(r"(?:^|[,;])\s*если\b", re.IGNORECASE)


def validate_command_form(text: str) -> None:
    wording = QUOTED_TITLE.sub("", text).strip() or text
    if "?" in wording or "？" in wording or CONDITIONAL_CLAUSE.search(wording):
        raise ValueError("Сформулируй прямую команду без вопросов и условий")
