"""Генерация каталога объектов для синтетических сценариев."""

import json
from math import ceil
from pathlib import Path

from ollama import ResponseError


BATCH_SIZE = 10


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


def validate_object(row: object, object_types: dict[str, str], ids: set[str], location: str) -> None:
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
            validate_object(row, object_types, ids, location)
            objects.append(row)
    if not objects:
        raise ValueError(f"{path}: мир пуст")
    return objects


def generate_world(
    object_types: dict[str, str], generate, *, n_objects_per_type: int, max_name_length: int,
    min_completion_ratio=0.8, max_generation_attempts=3,
) -> list[dict]:
    objects = []
    for object_type, description in object_types.items():
        names = {}
        empty_batches = 0
        last_problem = ""
        minimum = ceil(n_objects_per_type * min_completion_ratio)
        request_budget = ceil(n_objects_per_type / BATCH_SIZE) * max_generation_attempts
        for _ in range(request_budget):
            count = min(BATCH_SIZE, n_objects_per_type - len(names))
            schema = {
                "type": "array",
                "minItems": 1,
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
            if len(names) >= minimum:
                break

        if len(names) < minimum:
            print(
                f"Мир: {object_type} — получено {len(names)}/{n_objects_per_type}, "
                f"минимум {minimum}; {last_problem}", flush=True,
            )

        objects.extend(
            {"object_id": f"{object_type}_{index:06d}", "object_type": object_type, "name": name}
            for index, name in enumerate(names.values(), 1)
        )
    return objects
