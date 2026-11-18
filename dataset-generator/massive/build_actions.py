"""Export the MASSIVE-Agents function catalog as actions.jsonl."""

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError

from main import read_jsonl, write_lines


def build_actions(source):
    functions = {}
    for line_number, row in read_jsonl(source):
        catalog = row.get("function")
        if not isinstance(catalog, list) or not catalog:
            raise ValueError(f"{source}:{line_number}: missing function catalog")
        for function in catalog:
            name = function["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{source}:{line_number}: invalid function name {name!r}")
            if functions.setdefault(name, function) != function:
                raise ValueError(f"{source}:{line_number}: conflicting definitions of {name!r}")
    if not functions:
        raise ValueError(f"{source}: no functions found")

    actions = []
    for name, function in sorted(functions.items()):
        description = function["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{name}: missing function description")
        parameters = function["parameters"]
        if parameters["type"] not in ("dict", "object"):
            raise ValueError(f"{name}: unsupported parameters type {parameters['type']!r}")
        properties = {}
        for slot, spec in parameters["properties"].items():
            if not isinstance(spec.get("description"), str) or not spec["description"].strip():
                raise ValueError(f"{name}.{slot}: missing parameter description")
            properties[slot] = {key: value for key, value in spec.items() if key != "optional"}
        schema = {
            **parameters,
            "type": "object",
            "properties": properties,
            "required": parameters["required"],
            "additionalProperties": False,
        }
        Draft202012Validator.check_schema(schema)
        if set(schema["required"]) - properties.keys():
            raise ValueError(f"{name}: required parameters are missing from properties")
        actions.append({"action_id": name, "description": description, "parameters": schema})
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="MASSIVE-Agents input.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Destination actions.jsonl")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    try:
        actions = build_actions(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_lines(args.output, (json.dumps(action, ensure_ascii=False) for action in actions))
    except (OSError, ValueError, KeyError, TypeError, SchemaError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
