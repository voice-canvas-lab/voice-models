# Text dataset generator

The generator creates command templates with Ollama, fills them from a
synthetic world in Python, and uses Ollama to polish the spoken text.

## Status

The core generation stages are implemented and produce usable results in
pilot runs. The next step is to build a validator that checks whether the
final wording is natural and matches the labeled action and arguments.

## Setup and run

Start [Ollama](https://ollama.com/download), then run from the repository root:

```bash
ollama pull qwen2.5:3b
python3 -m venv .venv
.venv/bin/python -m pip install -r dataset-generator/requirements.txt
.venv/bin/python dataset-generator/text/main.py
```

- `--config PATH` loads another config. For a trial run, set a separate `run_dir`.
- `--world-only` regenerates the world. A normal run reuses the saved world or generates it if missing.
- `--use-prepared-examples` reuses arguments and context from the saved drafts with newly generated templates.
- If the proxy returns `502` for local Ollama, prefix the command with `NO_PROXY=localhost,127.0.0.1,::1`. Set `OLLAMA_HOST` to use another server.

## Files

`inputs_dir` contains `actions.jsonl` and `object_types.jsonl`. Action types
must exist in the object catalog. See the [data contracts](../../docs/contracts.md)
for schemas and `x-world-object-type` references.

A run writes these files under `run_dir`:

- `intermediate/world.jsonl` contains generated objects with IDs, types, and names.
- `intermediate/command_templates.jsonl` contains templates with named placeholders.
- `intermediate/draft_samples.jsonl` contains filled commands before speech editing.
- `dataset/samples.jsonl` contains final text, labels, context, IDs, and splits. TTS reads this file.

A normal run replaces the templates, drafts, and dataset. Reusing drafts preserves
the draft file.

## Configuration

The default is `configs/text.json`. Relative paths resolve from the repository root.

- `inputs_dir` and `run_dir` set the catalog and run directories.
- `n_objects_per_type` sets how many objects to generate **per type** when creating the world.
- `n_templates_per_action` and `n_texts_per_template` set the target number of templates per action and argument sets per template. Validation and deduplication can reduce the final count.
- `max_parameter_length` caps string arguments and object names in characters. Stricter action-schema limits still apply.
- `split_ratios` sets the fractions for train, validation, and test, in that order. The fractions must sum to 1.
- `seed` controls sampling and splits and is passed to Ollama.
- `model` selects the Ollama model. `temperature` controls sampling variability, `num_predict` caps generated tokens, and `num_ctx` sets the context window.
- `log_every` sets the progress interval in commands during speech editing.
