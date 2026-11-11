"""Генерация каталога объектов для синтетических сценариев."""

import json
from pathlib import Path

from ollama import ResponseError


BATCH_SIZE = 10
MAX_EMPTY_BATCHES = 3


def load_object_types(path: Path) -> dict[str, str]:
    object_types = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{location}: некорректная строка JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: ожидается описание типа объекта")
            object_type = row.get("object_type")
            description = row.get("description")
            if not isinstance(object_type, str) or object_type.split() != [object_type]:
                raise ValueError(f"{location}: object_type должен быть непустым идентификатором без пробелов")
            if object_type in object_types:
                raise ValueError(f"{location}: повторяется тип объекта {object_type!r}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{location}: отсутствует описание типа {object_type!r}")
            object_types[object_type] = description
    if not object_types:
        raise ValueError(f"{path}: каталог типов объектов пуст")
    return object_types


def load_world(path: Path, object_types: dict[str, str]) -> list[dict]:
    objects = []
    ids = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{location}: некорректная строка JSON") from exc
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("object_id", "object_type", "name")
            ):
                raise ValueError(f"{location}: ожидаются непустые object_id, object_type и name")
            if row["object_type"] not in object_types:
                raise ValueError(f"{location}: неизвестный тип {row['object_type']!r}")
            if row["object_id"] in ids:
                raise ValueError(f"{location}: повторяется object_id {row['object_id']!r}")
            if not isinstance(row.get("attributes", {}), dict) or row["name"].splitlines() != [row["name"]]:
                raise ValueError(f"{location}: ожидаются однострочное имя и объект attributes")
            ids.add(row["object_id"])
            objects.append(row)
    if not objects:
        raise ValueError(f"{path}: мир пуст")
    return objects


def generate_world(
    object_types: dict[str, str], generate, *, n_objects_per_type: int, max_name_length: int,
) -> list[dict]:
    objects = []
    for object_type, description in object_types.items():
        names = {}
        empty_batches = 0
        last_problem = ""
        while len(names) < n_objects_per_type:
            count = min(BATCH_SIZE, n_objects_per_type - len(names))
            schema = {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {"type": "string", "minLength": 1, "maxLength": max_name_length},
            }
            prompt = (
                f"Придумай {count} разных естественных заголовков для приложения. "
                f"Что называем: {description}. "
                "Дай каждому объекту один короткий законченный заголовок на подходящем языке. "
                "Числа, знаки и расширения файлов допустимы, если подходят объекту. "
                "Пиши только заголовок, тип объекта хранится отдельно. "
                f"Каждое название должно занимать одну строку и содержать до {max_name_length} символов. "
                "Верни JSON-массив строк."
            )
            if names:
                prompt += "\nУже есть, придумай другие: " + json.dumps(list(names.values()), ensure_ascii=False)
            if empty_batches:
                prompt += f"\nПовтор {empty_batches}: предложи другие допустимые названия."

            before = len(names)
            try:
                candidates = json.loads(generate(prompt, schema))
                if not isinstance(candidates, list):
                    raise ValueError("ожидался JSON-массив строк")
                for value in candidates:
                    if isinstance(value, str):
                        name = value.strip()
                        if 0 < len(name) <= max_name_length and name.splitlines() == [name]:
                            names.setdefault(name.casefold(), name)
                            if len(names) == n_objects_per_type:
                                break
                last_problem = "нет новых допустимых названий"
            except ResponseError as exc:
                if exc.status_code not in (408, 429, 500, 502, 503, 504):
                    raise
                last_problem = str(exc)
            except ValueError as exc:
                last_problem = str(exc)

            if len(names) > before:
                empty_batches = 0
                last_problem = ""
                print(f"Мир: {object_type} — {len(names)}/{n_objects_per_type}", flush=True)
            else:
                empty_batches += 1
                if empty_batches == MAX_EMPTY_BATCHES:
                    raise ValueError(
                        f"Мир: {object_type} — получено {len(names)}/{n_objects_per_type} названий; "
                        f"{last_problem}"
                    )

        objects.extend(
            {"object_id": f"{object_type}_{index:06d}", "object_type": object_type, "name": name}
            for index, name in enumerate(names.values(), 1)
        )
    return objects
