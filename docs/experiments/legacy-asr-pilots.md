# Historical ASR pilot notes

Archived README before publication cleanup. These are historical experiments, not the current setup or reproduction instructions. Some paths are machine-specific, outputs are local, and old four-clip metrics are not a model benchmark. Use [the current README](../../README.md) and [reproduction guide](../asr-reproduction.md).

---

# Voice Models

Models for converting Russian voice commands into structured actions for a notes and drawing application.

The planned pipeline is: **speech → text → actions**.

## Components

- **ASR:** adaptation of a pretrained speech recognition model to Russian voice commands.
- **Text to actions:** a BERT-based model for intent classification and parameter extraction, using application context when needed.

## Scope

This repository will contain model training, inference, data preparation, and evaluation code.

## Status

Early research and development. Model selection, datasets, and the action schema are being defined.

## Full pipeline: one command

From the repository root, with both existing environments prepared:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/pipeline-pilot.json
```

`configs/pipeline-pilot.json` runs the existing four texts with one `xenia` voice,
no noise and both ASR models. It configures the samples path, a **new** output
root, interpreter paths (`.venv` and `.venv-gigaam`), voices, SNR levels, seed,
Silero settings and model SHA256, and ASR model IDs, immutable revisions, CPU
thread counts and decoding settings. Paths inside the config are relative to the
repository root; absolute paths also work. Output must remain inside the repo.
For another run change `output`; completed and partial runs are never overwritten.
To enable more voices/noise, change `VOICES` and `NOISE_SNRS_DB` in this same file.

Before loading any models, the runner validates the full configuration and samples,
group/split consistency, both interpreter paths and installed required packages.
It rejects an existing output root. It installs nothing. Supported ASR models are
Whisper-small CPU INT8 and GigaAM-v3 e2e_ctc CPU float32; unsupported settings,
prompts, hotwords and mutable ASR revisions are rejected instead of ignored.
GigaAM uses its official greedy CTC decoder. Silero's configured SHA256 is checked
before loading its cached/downloaded package.

The runner invokes the existing scripts sequentially. The synthesis stage loads
Silero once, checks each clean WAV **before** creating its noise variants, then a
separate audio audit verifies the full dataset. Both ASR runs follow, then their
evaluations and the comparison report. This reuses the individual stage commands;
it does not introduce another implementation of synthesis, noise or metrics.

`outputs/pipeline-pilot/` contains requested/resolved configs, an immutable input
samples snapshot, environment versions, `status.json`, per-stage `logs/`,
`dataset/`, `whisper/`, `gigaam/`, `evaluation-whisper/`, `evaluation-gigaam/`
and `comparison/report.md`. Status is saved atomically before/after every stage.
On failure the runner stops with a nonzero exit code, marks the failed stage and
skips remaining stages. A run becomes `completed` only after all required stage
artifacts exist and ASR completion/audio-audit checks pass. Keep partial outputs
and logs for diagnosis; there is no automatic resume or overwrite.

The comparison reports corpus WER/CER and summed-time RTF overall and by
`action_id`, voice and acoustic condition. Every group includes WAV count,
unique `sample_id` count and unique `group_id` count; the report also states the
dataset's unique sample and paraphrase-group counts. Normalization and metric
definitions remain the same as in the individual evaluator.

## Optional lexical fragment retention

Manual examples are in `examples/fragment-pilot/samples.jsonl` (six complete
commands, empty context, validation split) and the separate
`examples/fragment-pilot/annotations.jsonl`. The samples schema is unchanged.
The sidecar contains 18 manually selected fragments: 10 names, 3 numbers and
5 negation/correction markers. It is never inferred from arbitrary `args`.
Examples without a single intended action use `action_id: null`, `args: {}`;
these labels are not used to judge the meaning of negative commands.

One JSONL row identifies a sample and its selected fragments, for example:

```json
{"sample_id":"fragment_003","fragments":[{"fragment_id":"command_negation","category":"negation","start_token":0,"end_token":1,"reference":"Не"}]}
```

`fragment_id` is unique within its sample. `category` is `name`, `number` or
`negation`; `reference` is the manually selected fragment. `start_token` is
inclusive and `end_token` exclusive, zero-based, **after the same normalization
as WER/CER**. Validation checks that the fragment matches these exact reference
tokens. Optional `variants` is a list of explicitly accepted nonempty strings.
For example, `двадцать одну` accepts `21` only when `"variants": ["21"]` is
present. There is no automatic number conversion, stemming or entity extraction.
Missing samples/categories in the sidecar are allowed; unknown sample IDs,
duplicate rows/fragment IDs, invalid spans or stale reference fragments fail.

Matching projects the marked reference occurrence through the global jiwer
minimum-edit word alignment and requires an exact **whole-token sequence** match
with the reference fragment or an explicit variant. It never searches for any
occurrence elsewhere: `Архив` does not match `архивный`, and a second `не` does
not rescue a deleted first `не`. Insertions inside the fragment are included.
Insertions at its outer boundaries are excluded, except when they complete an
explicit variant without consuming aligned neighboring reference tokens (for
example a single numeric token expressed as several words).

Limitations: equally good alignments of repeated text can be ambiguous; the
deterministic jiwer alignment is used. A substitution of unequal token lengths
that crosses an annotation boundary is conservatively marked
`ambiguous_alignment` and not preserved. The metric checks lexical retention,
not intent, negation scope, execution safety or the final corrected value.
In a self-correction, both the initial and corrected spoken numbers are annotated;
dropping the initial number or `нет` is a retention error even if the final number
is right. Here `нет` is explicitly labeled as a lexical negation/correction marker.

For each category the report gives checked fragments, preserved fragments and
`preserved / checked`. Every audio version contributes another check; these are
not independent commands. A category with no annotations has `retention: null`,
`status: "not_evaluated"` and is displayed as **«не оценено»**, never 0%.
`record_fragments.jsonl` keeps original and normalized full texts, each reference
span, accepted variants, projected hypothesis span, observed fragment and failure
reason. `summary.json` includes category totals, unique source-fragment/sample
counts, rules, jiwer version and input/code hashes; `report.md` includes errors.

Optional pipeline field:

```json
"annotations": "examples/fragment-pilot/annotations.jsonl"
```

`configs/pipeline-fragments.json` is prepared for this set. Omitting `annotations`
or setting it to `null` keeps the seven-stage pipeline and displays fragment
metrics as «не оценено». With a sidecar, preflight validates it before model
loading, then adds two evaluation-only stages after ordinary ASR evaluation and
includes their category tables in the joint report. The sidecar is snapshotted
at the run root. ASR/TTS receive `config.models.json`, which contains only audio
and model settings, **no annotation path or content**. Full requested/resolved
configurations remain separately available for provenance.

Standalone evaluation, **after** a matching dataset/ASR run exists:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_fragments.py --dataset outputs/pipeline-fragments/dataset --results outputs/pipeline-fragments/whisper --annotations examples/fragment-pilot/annotations.jsonl --output outputs/fragment-evaluation-whisper
```

For code checks only (no model loading or inference):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The new six-command set has not been synthesized or recognized; its annotation
and matching tests use explicit synthetic prediction fixtures only.

## Technical TTS pilot (Windows, CPU)

The four manually prepared examples in `examples/pilot/samples.jsonl` are a
technical probe, not the final dataset. They contain two groups of two folder
renaming commands, empty context, and only the `validation` split.
`available_action_ids` is omitted by team agreement; the shared contract is unchanged.

From the repository root, with the existing Python 3.11 `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tts.txt
.\.venv\Scripts\python.exe scripts/synthesize.py --samples examples/pilot/samples.jsonl --output outputs/pilot
```

The script follows the official [Silero standalone example](https://github.com/snakers4/silero-models#standalone-use).
It downloads `v5_ru.pt` once into `.cache/silero`, loads it once on CPU, and
synthesizes only each sample's `text` with `xenia` at 48 kHz. Polyphase filtering
resamples the signal to 16 kHz before writing mono PCM_16 WAV.
If a resampled peak reaches 0.999, gain is reduced to place the peak at 0.98
before PCM conversion; the gain is recorded in `audio_checks.json`.

The output root contains `samples.jsonl`, `audio/`, `audio_manifest.jsonl`,
`tts_settings.json` (model SHA256, versions and settings), and `audio_checks.json`.
Checks reopen every WAV and validate format, duration, finite/nonempty signal,
RMS and clipping. They do not replace listening for pronunciation and text agreement.
Existing output directories are rejected, including partial runs; choose a new
`--output` directory to retry. Output paths in the manifest are relative to that root.
Both `outputs/` and `.cache/` are ignored by Git. Keep the cached model and its hash
for reproduction because the upstream download URL is not an immutable revision.

The completed development probe is in `outputs/pilot-v2` (four WAV files).
`outputs/pilot` contains an earlier partial run that stopped on an excessive peak;
it was preserved. To listen to the completed probe, run:

```powershell
Invoke-Item outputs/pilot-v2/audio/audio_0001.wav
```

## Whisper pilot (Windows, CPU)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-asr.txt
.\.venv\Scripts\python.exe scripts/transcribe.py --dataset outputs/pilot-v2 --output outputs/asr-pilot
```

The script downloads `Systran/faster-whisper-small` into `.cache/huggingface`,
resolves the requested revision to a commit, and loads it once with CPU INT8.
Use `--revision <commit>` to reproduce that model version. Only WAV files are
passed to recognition; `samples.jsonl` and reference texts are never read.
Language is `ru`, task is `transcribe`, with no prompt or hotwords.

One complete warmup on the first WAV precedes measurements. Each measurement
includes audio reading and consumption of every generated segment, excluding
model loading, warmup and output writing. RTF uses duration read from the WAV.
The first measured file is therefore already warm in the filesystem cache.

A new output folder contains `predictions.jsonl` (`sample_id`, `audio_id`, `text`),
`timings.jsonl` (time, duration, RTF and WAV hash), and `asr_settings.json`
(model revision, effective decoding settings, installed dependencies, CPU and OS).
Existing output directories are rejected, including partial runs. Choose a new
output directory for every run. Compare references by `sample_id` only after
recognition; these four validation examples are a technical probe, not a benchmark.

## Evaluate ASR

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-asr.txt
.\.venv\Scripts\python.exe scripts/evaluate.py --dataset outputs/pilot-v2 --results outputs/asr-pilot --output outputs/evaluation-pilot
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Evaluation uses the dataset's snapshot `samples.jsonl`, joins predictions through
`audio_manifest.jsonl`, and requires complete matching `timings.jsonl`. Duplicate
IDs, missing/extra predictions or timings, unknown samples and mismatched IDs
cause errors. Each sample may have multiple audio IDs.

Both texts receive Unicode NFC, lowercase, `ё` → `е`, Unicode punctuation
replacement with spaces and whitespace collapse. Words, numbers, negations and
self-corrections remain. CER counts spaces in the normalized text.
Empty normalized references are retained and any hypothesis units count as
insertions. Rates with a zero reference denominator are `null` (undefined),
including empty/empty pairs; error counts remain available and enter corpus sums.

A new output folder contains `record_metrics.jsonl` with original and normalized
texts, word/character edit counts and rates; `summary.json` with corpus rates,
overall RTF, input hashes and jiwer version; and a readable `report.md`.
Corpus WER/CER divide summed errors by summed reference words/characters.
Overall RTF divides summed recognition time by summed audio duration.
Existing output folders are rejected. Inputs are read only.

## GigaAM-v3 e2e_ctc pilot (Windows, CPU)

Use a separate environment: the official HF model card recommends newer
PyTorch/torchaudio than the TTS environment. The existing `.venv` is unchanged.

```powershell
& 'C:\Users\kisly\python.exe' -m venv .venv-gigaam
.\.venv-gigaam\Scripts\python.exe -m pip install -r requirements-gigaam.txt
.\.venv-gigaam\Scripts\python.exe scripts/transcribe_gigaam.py --dataset outputs/pilot-v2 --output outputs/gigaam-pilot-v2
.\.venv\Scripts\python.exe scripts/evaluate.py --dataset outputs/pilot-v2 --results outputs/gigaam-pilot-v2 --output outputs/evaluation-gigaam-pilot
```

The implementation follows the official [GigaAM HF example](https://github.com/salute-developers/GigaAM#loading-from-hugging-face)
and [model card](https://huggingface.co/ai-sage/GigaAM-v3).
It pins the `e2e_ctc` branch commit `cec030b4c4f35d928e4a9044a3bdb29ebd499fac`
and checks `model_name=v3_e2e_ctc` before loading the official remote code.
CPU inference uses float32 and greedy CTC decoding, four threads, one model load
and one complete warmup. Only WAV paths enter recognition; references are not read.
Short-form inference rejects audio over 25 seconds. No long-form/VAD dependencies
are installed; `pyannote.core` supplies the `pyannote` namespace required by
Transformers' static scan of the optional long-form imports. Short-form inference
does not call `pyannote.audio`. FFmpeg is bundled via `imageio-ffmpeg` and copied into the ignored
project cache for the official audio loader.

Output formats match Whisper: `predictions.jsonl`, `timings.jsonl`,
`asr_settings.json`. Timings include FFmpeg reading and obtaining the complete
text, but exclude loading and warmup. Model revision, decoding configuration,
dependencies and hardware are recorded separately. Existing output roots are
rejected. `.venv-gigaam/`, `.cache/` and `outputs/` are ignored by Git.
Use the same evaluator and normalization rules for both models. Compare CPU
float32 GigaAM with CPU INT8 Whisper as these actual configurations; four
synthetic validation clips are only a technical probe, not a quality benchmark.

Completed local results are in `outputs/gigaam-pilot-v2`, evaluation in
`outputs/evaluation-gigaam-pilot`, and the joint table with original transcripts
and errors in `outputs/comparison-whisper-gigaam/report.md`. The earlier empty
`outputs/gigaam-pilot` was preserved after Transformers rejected a missing
`pyannote` namespace. Select new output names when rerunning these commands.

| Model | Corpus WER | Corpus CER | Overall RTF |
|---|---:|---:|---:|
| Whisper-small, CPU INT8 | 4.55% | 1.32% | 0.918 |
| GigaAM-v3 e2e_ctc, CPU float32 | 0% | 0% | 0.154 |

Whisper substituted `Переменой` for `Переименуй` in `audio_0003`.
GigaAM had no errors after normalization on these four clips. The models used
the same WAV hashes and four CPU threads, but different runtimes, precision and
audio decoders. Timings were obtained in separate runs, not a controlled speed
benchmark. This tiny synthetic set cannot establish general model quality.

## Multiple voices and artificial noise

`configs/pilot-multivoice.json` selects `aidar`, `baya`, `xenia`,
`NOISE_SNRS_DB: [20, 10]`, and `SEED: 42`. JSON is used so no YAML dependency is
needed. `configs/pilot-clean.json` keeps one `xenia` voice and no noise; it is the
default when `--config` is omitted. Both modes load Silero once and pass only
each sample's unchanged `text` to TTS. The dataset keeps the original samples,
groups and splits. Each WAV has a unique ID, voice, split and synthesis seed;
each noise variant points to its clean `source_audio_id`.

This technical probe uses **artificial white Gaussian noise, not recordings of
real acoustic environments**. Power is `mean(x**2)` over **all** decoded clean
PCM_16 samples, including pauses, without VAD, centering or DC removal. For raw
standard-normal noise `z`, its scale is
`sqrt(mean(clean**2) / (mean(z**2) * 10**(SNR/10)))`.
Peak protection scales the **whole mixture** to a peak of 0.98 if needed,
preserving the signal/noise ratio. Each variant records its stable per-file
PCG64 seed (derived from seed/sample/voice/SNR), noise scale and mixture gain.

The actual post-PCM SNR is
`10*log10(mean((gain*clean_pcm)**2) / mean((noisy_pcm-gain*clean_pcm)**2))`.
Thus the residual includes quantization error. Tolerance is 0.05 dB.
The audit reopens every WAV, checks IDs, coverage, source links, splits, duration,
nonempty signal and clipping; it also regenerates every noise waveform from its
seed and checks bit-exact PCM equality. Metadata fixes the TTS model hash,
configuration, power/SNR rules and software versions.

```powershell
.\.venv\Scripts\python.exe scripts/synthesize.py --samples examples/pilot/samples.jsonl --config configs/pilot-multivoice.json --output outputs/pilot-multivoice
.\.venv\Scripts\python.exe scripts/check_dataset.py --dataset outputs/pilot-multivoice --output outputs/pilot-multivoice/dataset_checks.json
.\.venv\Scripts\python.exe scripts/transcribe.py --dataset outputs/pilot-multivoice --output outputs/asr-multivoice-whisper --revision 536b0662742c02347bc0e980a01041f333bce120
.\.venv-gigaam\Scripts\python.exe scripts/transcribe_gigaam.py --dataset outputs/pilot-multivoice --output outputs/asr-multivoice-gigaam
.\.venv\Scripts\python.exe scripts/evaluate.py --dataset outputs/pilot-multivoice --results outputs/asr-multivoice-whisper --output outputs/evaluation-multivoice-whisper
.\.venv\Scripts\python.exe scripts/evaluate.py --dataset outputs/pilot-multivoice --results outputs/asr-multivoice-gigaam --output outputs/evaluation-multivoice-gigaam
.\.venv\Scripts\python.exe scripts/compare_conditions.py --dataset outputs/pilot-multivoice --whisper outputs/evaluation-multivoice-whisper --gigaam outputs/evaluation-multivoice-gigaam --output outputs/comparison-multivoice
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run ASR commands sequentially to avoid simultaneous CPU load. Choose fresh output
names for repeat runs; existing folders and audit files are never overwritten.
The four-sample configuration produces 12 clean + 24 noisy = 36 WAV files.
The comparison report includes overall, per-voice and per-condition corpus
WER/CER and RTF, sample counts, edit totals, audio checks and error examples.
Group rates divide summed errors by summed reference units, not mean percentages.
Per-voice groups contain 12 recordings each (all conditions); per-condition groups
contain 12 recordings each (all voices), with the same four text samples reused.
These are correlated synthetic recordings, not 36 independent commands.
