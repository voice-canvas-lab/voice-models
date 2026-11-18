"""Подбор аргументов из мира и подстановка в шаблоны."""

import json
import re
from copy import deepcopy
from random import Random

from jsonschema import Draft202012Validator


def prepare_examples(
    action: dict,
    templates: list[dict],
    count: int,
    *,
    world: list[dict],
    rng: Random,
    max_parameter_length: int,
    prepared_examples=None,
) -> list[dict]:
    schema = deepcopy(action["parameters"])
    for property_schema in schema["properties"].values():
        if property_schema["type"] == "string":
            property_schema["maxLength"] = min(
                property_schema.get("maxLength", max_parameter_length), max_parameter_length,
            )
    validator = Draft202012Validator(schema)
    arguments_by_fields = {}
    examples = []
    for template in templates:
        fields = tuple(sorted(template["parameters"]))
        if fields not in arguments_by_fields:
            if prepared_examples is None:
                arguments_by_fields[fields] = _sample_scenarios(
                    action["action_id"], fields, schema["properties"], validator, world, rng, count,
                )
            else:
                arguments_by_fields[fields] = _reuse_scenarios(
                    prepared_examples, action["action_id"], fields, validator,
                )
        for example in arguments_by_fields[fields]:
            values = dict(example["args"])
            for field, value in values.items():
                property_schema = schema["properties"][field]
                if isinstance(value, str) and ("enum" in property_schema or "const" in property_schema):
                    label = re.search(
                        rf"\b{re.escape(value)}\s*[—–-]\s*([^,;.]+)", property_schema["description"],
                    )
                    if label:
                        values[field] = label.group(1).strip()
            examples.append({**example, "text": template["text"].format(**values)})
    return examples


def _sample_scenarios(action_id, fields, properties, validator, world, rng, count):
    slot_validators = {field: Draft202012Validator(properties[field]) for field in fields}
    type_validator = Draft202012Validator(properties.get("object_type", {}))
    reference_fields = [field for field in fields if "x-world-object-type" in properties[field]]
    source_fields = [
        field for field in fields
        if field not in reference_fields and field in ("name", "old_name", "query", "object_id")
    ]
    uses_subject = source_fields or "object_type" in fields or "new_name" in fields or action_id == "save"
    subjects = [
        obj for obj in world
        if type_validator.is_valid(obj["object_type"])
        and all(
            slot_validators[field].is_valid(obj["object_id" if field == "object_id" else "name"])
            for field in source_fields
        )
    ] if uses_subject else [None]
    rng.shuffle(subjects)

    reference_pools = {
        field: [
            obj for obj in world
            if obj["object_type"] == properties[field]["x-world-object-type"]
            and slot_validators[field].is_valid(obj["name"])
        ]
        for field in reference_fields
    }
    other_pools = {}
    for field in fields:
        if field in source_fields or field in reference_fields or field in ("object_type", "new_name"):
            continue
        prop = properties[field]
        if "const" in prop:
            candidates = [prop["const"]]
        elif "enum" in prop:
            candidates = prop["enum"]
        else:
            candidates = list(dict.fromkeys(obj["name"] for obj in world))
        other_pools[field] = [value for value in candidates if slot_validators[field].is_valid(value)]

    unique = {}
    if subjects and all(reference_pools.values()) and all(other_pools.values()):
        for attempt in range(max(len(subjects), count * 10)):
            subject = subjects[attempt % len(subjects)]
            args = {field: rng.choice(pool) for field, pool in other_pools.items()}
            objects = []
            reserved_names = set()
            if subject is not None:
                args.update({
                    field: subject["object_id" if field == "object_id" else "name"]
                    for field in source_fields
                })
                if "object_type" in fields:
                    args["object_type"] = subject["object_type"]
                reserved_names.add(subject["name"].casefold())
                if action_id == "save" and "name" not in fields:
                    objects.append({
                        **subject, "attributes": {**subject.get("attributes", {}), "is_open": True},
                    })
                elif source_fields and action_id != "create":
                    objects.append(subject)

            for field, pool in reference_pools.items():
                candidates = [obj for obj in pool if obj["name"].casefold() not in reserved_names]
                if candidates:
                    reference = rng.choice(candidates)
                    args[field] = reference["name"]
                    objects.append(reference)
                    reserved_names.add(reference["name"].casefold())

            if "new_name" in fields:
                names = [
                    obj["name"] for obj in world
                    if obj["object_type"] == subject["object_type"]
                    and obj["name"].casefold() not in reserved_names
                    and slot_validators["new_name"].is_valid(obj["name"])
                ]
                if names:
                    args["new_name"] = rng.choice(names)

            if len(args) != len(fields) or not validator.is_valid(args):
                continue
            scenario = {
                "action_id": action_id,
                "args": args,
                "context": {"objects": objects, "selected_object_ids": []},
            }
            unique.setdefault(json.dumps(scenario, ensure_ascii=False, sort_keys=True), scenario)
            if len(unique) == count:
                break

    if len(unique) < count:
        slots = ", ".join(fields) or "без параметров"
        print(f"{action_id} ({slots}): собрано {len(unique)}/{count} наборов аргументов.", flush=True)
    return list(unique.values())


def _reuse_scenarios(examples, action_id, fields, validator):
    unique = {}
    for example in examples:
        if (
            example["action_id"] == action_id
            and tuple(sorted(example["args"])) == fields
            and validator.is_valid(example["args"])
        ):
            unique.setdefault(json.dumps(example, ensure_ascii=False, sort_keys=True), example)
    return list(unique.values())
