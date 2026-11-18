"""Convert the Russian MASSIVE pilot to the project's text dataset format."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from jsonschema import Draft202012Validator


DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "v1.1"
LOCALE = "ru-RU"
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
SLOT = re.compile(r"\[([a-z][a-z0-9_]*)\s*:\s*([^\[\]]+?)\s*\]")
logger = logging.getLogger(__name__)


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield line_number, row


def parse_args(text, annotated_text):
    if not isinstance(text, str) or not text.strip() or not isinstance(annotated_text, str):
        raise ValueError("missing or invalid utt/annot_utt")
    if SLOT.sub(lambda match: match[2], annotated_text) != text:
        raise ValueError("annot_utt does not reconstruct utt")

    args = {}
    for match in SLOT.finditer(annotated_text):
        slot, value = match.groups()
        if not value.strip():
            raise ValueError(f"empty slot {slot!r}")
        if slot in args:
            raise ValueError(f"repeated slot {slot!r}")
        args[slot] = value
    return args


def convert(source, catalog):
    validators = {}
    for _, action in read_jsonl(catalog):
        action_id = action["action_id"]
        if action_id in validators:
            raise ValueError(f"duplicate catalog action {action_id!r}")
        schema = action["parameters"]
        Draft202012Validator.check_schema(schema)
        validators[action_id] = Draft202012Validator(schema)

    samples = []
    rejected = []
    counts = {action_id: dict.fromkeys(SPLITS.values(), 0) for action_id in validators}
    outside_pilot = Counter()
    seen_ids = set()
    for line_number, row in read_jsonl(source):
        if row.get("locale") != LOCALE:
            continue
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: invalid or duplicate MASSIVE id {sample_id!r}")
        seen_ids.add(sample_id)
        action_id = row.get("intent")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"{source}:{line_number}: invalid MASSIVE intent {action_id!r}")
        if action_id not in validators:
            outside_pilot[action_id] += 1
            continue

        try:
            partition = row.get("partition")
            if not isinstance(partition, str) or partition not in SPLITS:
                raise ValueError(f"unknown partition {partition!r}")
            split = SPLITS[partition]
            text = row.get("utt")
            args = parse_args(text, row.get("annot_utt"))
            validator = validators[action_id]
            unknown_slots = sorted(args.keys() - validator.schema["properties"].keys())
            if unknown_slots:
                raise ValueError("slots outside the pilot catalog: " + ", ".join(unknown_slots))
            error = next(validator.iter_errors(args), None)
            if error is not None:
                raise ValueError(error.message)
        except ValueError as exc:
            rejected.append({"id": sample_id, "intent": action_id, "line": line_number, "reason": str(exc)})
            continue

        samples.append({
            "sample_id": sample_id,
            "text": text,
            "context": {"objects": [], "selected_object_ids": []},
            "action_id": action_id,
            "args": args,
            "group_id": sample_id,
            "split": split,
        })
        counts[action_id][split] += 1

    return samples, {
        "locale": LOCALE,
        "imported": len(samples),
        "counts": counts,
        "outside_pilot": dict(sorted(outside_pilot.items())),
        "rejected": rejected,
    }


def write_lines(path, lines):
    output = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(output.name)
    try:
        with output:
            for line in lines:
                output.write(line + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local MASSIVE ru-RU JSONL")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "dataset")
    args = parser.parse_args()

    samples_path = args.output_dir / "samples.jsonl"
    if args.input.resolve() == samples_path.resolve():
        parser.error("--input must be different from the output file")
    logging.basicConfig(format="%(levelname)s: %(message)s")
    try:
        samples, report = convert(args.input, DATASET_DIR / "inputs" / "actions.jsonl")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    for row in report["rejected"]:
        logger.warning(
            "MASSIVE line %s, id=%s (%s): %s", row["line"], row["id"], row["intent"], row["reason"],
        )
    if not samples:
        parser.exit(1, "No pilot samples imported.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(samples_path, (json.dumps(sample, ensure_ascii=False) for sample in samples))
    print(f"Imported {len(samples)} samples, rejected {len(report['rejected'])}.")


if __name__ == "__main__":
    main()
