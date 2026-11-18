"""Export compatible Russian MASSIVE examples as samples.jsonl."""

import argparse
import json
import logging
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError

from main import LOCALE, SPLITS, parse_args, read_jsonl, write_lines


logger = logging.getLogger(__name__)


def read_catalog(path):
    validators = {}
    for line_number, action in read_jsonl(path):
        name = action["action_id"]
        if not isinstance(name, str) or not name.strip() or name in validators:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate action_id {name!r}")
        schema = action["parameters"]
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(schema)
    if not validators:
        raise ValueError(f"{path}: empty action catalog")
    return validators


def register_id(path, line_number, row, seen_ids):
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
        raise ValueError(f"{path}:{line_number}: invalid or duplicate source id {sample_id!r}")
    seen_ids.add(sample_id)
    return sample_id


def make_sample(sample_id, text, action_id, args, split, validators):
    if action_id not in validators:
        raise ValueError(f"action {action_id!r} is absent from the catalog")
    validator = validators[action_id]
    error = next(validator.iter_errors(args), None)
    if error is not None:
        raise ValueError(error.message)
    return {
        "sample_id": sample_id,
        "text": text,
        "context": {"objects": [], "selected_object_ids": []},
        "action_id": action_id,
        "args": args,
        "group_id": sample_id,
        "split": split,
    }


def parse_gold_args(gold_args, schema):
    if not isinstance(gold_args, dict):
        raise ValueError("expected an object of gold arguments")
    args = {}
    properties = schema["properties"]
    for name, options in gold_args.items():
        if name not in properties:
            raise ValueError(f"unknown gold argument {name!r}")
        if not isinstance(options, list) or not options or any(not isinstance(value, str) for value in options):
            raise ValueError(f"{name}: expected a nonempty list of gold values")
        values = set(options)
        if "" in values:
            values.remove("")
            if values and values != {properties[name].get("default")}:
                raise ValueError(f"{name}: omission and a non-default value are both accepted")
        elif len(values) == 1:
            args[name] = next(iter(values))
        else:
            raise ValueError(f"{name}: multiple gold values")
    return args


def build_samples(source, validators, agents_test=None):
    samples = []
    skipped = 0
    seen_ids = set()
    source_test = {}
    for line_number, row in read_jsonl(source):
        if row.get("locale") != LOCALE:
            continue
        sample_id = register_id(source, line_number, row, seen_ids)
        intent = row.get("intent")
        partition = row.get("partition")
        text = row.get("utt")
        if agents_test is not None and partition == "test":
            source_test[sample_id] = text
            continue
        try:
            if not isinstance(partition, str) or partition not in SPLITS:
                raise ValueError(f"unknown partition {partition!r}")
            if not isinstance(intent, str) or "_" not in intent:
                raise ValueError(f"invalid intent {intent!r}")
            action_id = intent.replace("_", ".", 1)
            args = parse_args(text, row.get("annot_utt"))
            samples.append(make_sample(sample_id, text, action_id, args, SPLITS[partition], validators))
        except ValueError as exc:
            logger.warning("MASSIVE line %s, id=%s: %s", line_number, sample_id, exc)
            skipped += 1

    if agents_test is not None:
        test_ids = set()
        for line_number, row in read_jsonl(agents_test):
            sample_id = register_id(agents_test, line_number, row, test_ids)
            try:
                text = row.get("question")
                if not isinstance(text, str) or not text.strip() or text != source_test.get(sample_id):
                    raise ValueError("id and question do not match a Russian MASSIVE test row")
                gold = row.get("gt")
                if not isinstance(gold, dict) or len(gold) != 1:
                    raise ValueError("expected exactly one gold function")
                (action_id, gold_args), = gold.items()
                if action_id not in validators:
                    raise ValueError(f"action {action_id!r} is absent from the catalog")
                args = parse_gold_args(gold_args, validators[action_id].schema)
                samples.append(make_sample(sample_id, text, action_id, args, "test", validators))
            except ValueError as exc:
                logger.warning("MASSIVE-Agents line %s, id=%s: %s", line_number, sample_id, exc)
                skipped += 1
    return samples, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original MASSIVE ru-RU.jsonl")
    parser.add_argument("--actions", required=True, type=Path, help="Catalog from build_actions.py")
    parser.add_argument("--output", required=True, type=Path, help="Destination samples.jsonl")
    parser.add_argument("--agents-test", type=Path, help="Replace the test split with MASSIVE-Agents input.jsonl")
    args = parser.parse_args()
    input_paths = {path.resolve() for path in (args.input, args.actions, args.agents_test) if path is not None}
    if args.output.resolve() in input_paths:
        parser.error("--output must be different from the input files")
    logging.basicConfig(format="%(levelname)s: %(message)s")
    try:
        samples, skipped = build_samples(args.input, read_catalog(args.actions), args.agents_test)
        if not samples:
            raise ValueError("no compatible samples found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_lines(args.output, (json.dumps(sample, ensure_ascii=False) for sample in samples))
    except (OSError, ValueError, KeyError, TypeError, SchemaError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Imported {len(samples)} samples, skipped {skipped}.")


if __name__ == "__main__":
    main()
