# MASSIVE text adapter

Dataset version `data/v1.1/` imports the `ru-RU` subset of MASSIVE 1.1 using
the function catalog from MASSIVE-Agents. Amazon distributes both sources under
CC BY 4.0. [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive)
· [MASSIVE-Agents](https://huggingface.co/datasets/AmazonScience/massive-agents)

Run these commands from the repository root with `jsonschema` installed. Keep
the [MASSIVE archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz)
in `data/v1.1/raw/massive/` and the Russian
[MASSIVE-Agents input file](https://huggingface.co/datasets/AmazonScience/massive-agents/tree/b6156972182bdf34e68c5b5dfbfe6d30db82f104/massive-full-converted-all-langs-with-id/ru-RU)
in `data/v1.1/raw/massive-agents/`.

```bash
mkdir -p data/v1.1/raw/massive data/v1.1/raw/massive-agents
tar -xOf data/v1.1/raw/massive/amazon-massive-dataset-1.1.tar.gz 1.1/data/ru-RU.jsonl \
  > data/v1.1/raw/massive/ru-RU.jsonl
```

Build the catalog and samples after placing both source files in those directories:

```bash
.venv/bin/python dataset-generator/massive/build_actions.py \
  --input data/v1.1/raw/massive-agents/input.jsonl \
  --output data/v1.1/inputs/actions.jsonl
.venv/bin/python dataset-generator/massive/build_samples.py \
  --input data/v1.1/raw/massive/ru-RU.jsonl \
  --actions data/v1.1/inputs/actions.jsonl \
  --output data/v1.1/dataset/samples.jsonl
```

Add `--agents-test data/v1.1/raw/massive-agents/input.jsonl` to the second command
to use MASSIVE-Agents for the test split. The import logs the source ID and
reason for each incompatible record to stderr.

## Output format

- `build_actions.py` converts the source function definitions to the project's
  [action contract](../../docs/contracts.md).
- `build_samples.py` preserves the source `id` as `sample_id` and maps intent
  labels such as `weather_query` to catalog names such as `weather.query`.
  The script validates arguments against the catalog.
  With `--agents-test`, test examples use the reference calls from
  MASSIVE-Agents. The `dev` split is exported as `validation`, and each example
  uses its source ID as a provisional `group_id`.


## Status

**Limitation — argument normalization.** The adapter stores source values as
written. For example, `семь утра` stays a string. Conversion to a canonical
value such as `07:00` requires a separate step.
