# Text dataset generator

The generator creates command templates with Ollama, fills them from a
synthetic world in Python, and uses Ollama for local grammatical corrections.
The speech-editing prompt asks the model to preserve template structure, word order, and
argument roles while allowing case, agreement, preposition, and punctuation
corrections. It requires exact preservation of numbers, dates, versions,
identifiers, and file extensions. These constraints are prompt
instructions; the current validator checks response and command format.

## Status

**Use of the text generator is temporarily frozen because the generated data
has insufficient quality.** Observed problems include unnatural wording,
incorrect verb forms, distorted action meanings, and changes to argument
values during speech editing. The generator and `data/v1/` are retained as
experimental artifacts for reproduction and error analysis.

As of 2026-11-17, the working v1 files contain 80 world objects, 36 command
templates, 170 drafts, and 140 final samples. The catalog and templates cover
eight actions. Drafts and final samples cover seven, with `move` absent.
These counts describe the saved files; they are not guaranteed output sizes.

The planned replacement is our dataset version `data/v1.1/`, based on MASSIVE.
Its adapter and preparation are pending.

## Validation and limitations

- Catalogs, argument schemas, and world objects are validated. Prepared drafts
  are checked before template generation, including required context object
  fields, unique object IDs, known object types, and selection references.
- Templates are checked for required and allowed placeholders. Command-form
  checks reject question marks outside quoted titles and some conditional
  constructions. Russian grammar and imperative verb forms are not validated.
- Speech editing checks response structure and command form. Preservation of
  argument values, word order, and action meaning relies on prompt instructions.
  A validator for semantic agreement between `text`, `args`, and `context`
  remains pending.
- Deduplication uses text, action, arguments, and context together. Splits group
  identical action/argument/context combinations. Identical text with different
  labels or context can occur across splits.

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
- `--use-prepared-examples` validates the saved drafts before generating new templates, then reuses their arguments and context.
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

A normal run writes templates, drafts, and the dataset as their stages complete.
Each file is replaced atomically, but a failure in a later stage can leave files
from different runs. Reusing drafts preserves the draft file and regenerates
templates and final text through Ollama.

## Configuration

The default is `configs/text.json`. Relative paths resolve from the repository root.

- `inputs_dir` and `run_dir` set the catalog and run directories.
- `n_objects_per_type` sets how many objects to generate **per type** when creating the world.
- `n_templates_per_action` sets the target number of templates per action. In a normal run, `n_texts_per_template` sets the target number of argument sets per template. With `--use-prepared-examples`, each template uses all distinct saved scenarios that match its action and parameter names and pass validation. Validation and deduplication can reduce the final count.
- `max_parameter_length` caps string arguments and object names in characters. Stricter action-schema limits still apply.
- `split_ratios` sets the fractions for train, validation, and test, in that order. The fractions must sum to 1.
- `seed` controls sampling and splits and is passed to Ollama.
- `model` selects the Ollama model. `temperature` controls sampling variability, `num_predict` caps generated tokens, and `num_ctx` sets the context window.
- `log_every` sets the progress interval in commands during speech editing.

## Generation completion and retries

- `min_completion_ratio` (default 0.8) sets the minimum accepted fraction,
  rounded up. It applies per object type in world generation, per action in
  template generation, and per batch of up to four commands in speech editing.
- `max_generation_attempts` (default 3) caps requests per action or speech batch.
  World generation uses batches of up to ten names, with a request budget of
  `ceil(n_objects_per_type / 10) * max_generation_attempts` per object type.
- Accepted results accumulate across requests. Generation stops at the minimum.
  If the budget is exhausted, accepted results are retained and a shortfall is
  logged. Empty results still trigger the pipeline's existing stage checks.
- Existing world files and prepared drafts are reused. Argument sampling,
  deduplication, and splitting are Python operations and do not request an LLM.
  These settings do not supply missing actions in prepared drafts.
- Completion measures acceptance by the existing validators. Semantic agreement
  and linguistic quality still require the checks described above.
