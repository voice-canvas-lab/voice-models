# Synthetic text generator

Builds labeled Russian voice commands for the actions in `actions.jsonl`.
Ollama generates command templates and edits the filled commands for speech.
Python fills the templates with arguments from a synthetic world and assigns
the dataset splits.

Use is **temporarily frozen** because some generated commands are unnatural
or disagree with their labels. The synthetic dataset is in `data/v1/`.
The current dataset uses the [MASSIVE adapter](../massive/README.md).

## Run

Start [Ollama](https://ollama.com/download), then run from the repository root.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r dataset-generator/requirements.txt
ollama pull qwen2.5:3b
.venv/bin/python dataset-generator/text/main.py
```

The default configuration is [configs/text.json](../../configs/text.json).
Use `--config PATH` with a separate `run_dir` for a new run. The default run
writes to `data/v1/` and reuses `world.jsonl` if it exists.

- `--world-only` regenerates the world and stops.
- `--use-prepared-examples` reuses arguments and context from saved drafts
  with newly generated templates.

## Configuration

### Paths

- `inputs_dir` — input catalogs.
- `run_dir` — output directory.

### Dataset size and splits

- `n_objects_per_type` — target number of world objects per type.
- `n_templates_per_action` — target number of templates per action.
- `n_texts_per_template` — target number of argument sets per template.
- `max_parameter_length` — character limit for string arguments and generated names.
- `split_ratios` — train, validation, and test proportions, in that order.

### Ollama

- `model` — model name.
- `temperature` — sampling variability.
- `num_predict` — maximum response tokens.
- `num_ctx` — context window in tokens.
- `seed` — random seed for Ollama, argument sampling, and dataset splitting.

### Retries and progress

- `min_completion_ratio` — fraction of accepted results needed to stop retrying.
- `max_generation_attempts` — request budget, scaled by the number of batches
  when generating the world.
- `log_every` — number of processed commands between speech-editing progress updates.

## Data

- `inputs/` contains the action and object-type catalogs.
- `intermediate/` contains the world, command templates, and filled drafts.
- `dataset/samples.jsonl` contains the final commands, action labels, arguments,
  context, and split assignments.

The fields are defined in [the data contracts](../../docs/contracts.md).
