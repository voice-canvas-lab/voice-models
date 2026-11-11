# Voice Models

Russian voice commands for a notes and drawing application. The planned product pipeline is **speech → text → actions**. The implemented research path here is **TTS → audio audit → artificial noise → ASR → evaluation**. Text-to-action modeling is separate; these ASR experiments do not measure action execution or repair action labels.

## Current reference experiment

[Zhanna ru-text-only-1.0.1 report](docs/reports/zhanna-text-only-v1.0.1.md): 151 of 163 commands, three Silero voices, clean/SNR20/SNR10, 1359 WAV, Whisper-small and GigaAM-v3 e2e_ctc. This is a **preliminary evaluation on synthetic speech**. Five confirmed pilot WAV do not validate the full dataset. Twelve unresolved pronunciation cases remain separate; labels, context and original texts were not repaired.

The report includes model revisions, hardware, ordinary and numerical WER/CER, preliminary RTF, slices, input hashes, and examples. [Six recordings for listening](docs/reports/zhanna-text-only-v1.0.1/review-six.md) have exact references, hypotheses and PowerShell commands. Audio stays in local ignored `outputs/`; compact text results are versioned under `docs/reports/`.

Synthetic audio does not always exactly match its text reference. The user heard digit repetitions in one reviewed recording, with their exact count and correspondence to the reference unresolved; in another recording the user heard no repetitions although Whisper produced a long repeated tail. These observations concern only those two WAV; see the [report and hash-bound listening observations](docs/reports/zhanna-text-only-v1.0.1.md#примеры-и-ручная-проверка).

## Install two environments (Windows, Python 3.11, CPU)

Run from the repository root. Use Python 3.11 with the Windows `py` launcher, or substitute the path to your own Python 3.11 executable. No activation is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tts.txt -r requirements-asr.txt

py -3.11 -m venv .venv-gigaam
.\.venv-gigaam\Scripts\python.exe -m pip install -r requirements-gigaam.txt

.\.venv\Scripts\python.exe -m pip check
.\.venv-gigaam\Scripts\python.exe -m pip check
```

`.venv` contains TTS, Whisper and evaluation. `.venv-gigaam` isolates the newer CPU PyTorch/torchaudio required by GigaAM. Do not install the two sets of torch pins into one environment. FFmpeg for GigaAM is supplied by `imageio-ffmpeg`; its project-local binary is cached automatically. First model runs require downloads; caches, weights and environments are not committed.

## Checks and evaluation without models

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests validate contracts, nonempty context, group consistency, noise, frozen pronunciation behavior, numerical equivalence, and evaluator failure handling. They do not download or load models. Some tests intentionally print pipeline failures using test fixtures.

Recompute metrics from the published predictions and timings without any WAV or model download:

```powershell
$Artifacts = 'docs/reports/zhanna-text-only-v1.0.1/artifacts'
$Check = 'outputs/published-evaluation-check' # must be a new directory
if (Test-Path -LiteralPath $Check) { throw 'Choose a new output directory' }
foreach ($Model in @('whisper', 'gigaam')) {
    .\.venv\Scripts\python.exe -B scripts/evaluate.py --dataset "$Artifacts/dataset" --results "$Artifacts/$Model" --output "$Check/evaluation-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    .\.venv\Scripts\python.exe -B scripts/evaluate_numeric.py --dataset "$Artifacts/dataset" --results "$Artifacts/$Model" --baseline "$Check/evaluation-$Model" --output "$Check/numeric-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Numerical evaluation failed' }
}
```

The published metadata bundle intentionally contains no WAV. These commands verify evaluation reproducibility, not acoustic correctness. Audio audit requires the local WAV or regenerated audio.

## Reproduce audio and ASR

Follow [the stage-by-stage guide](docs/asr-reproduction.md). It uses the preserved 151 references and pinned configuration, audits the audio, and checks **every WAV against the GigaAM 25-second limit before either full ASR**. It also runs the separate numerical evaluation. Every rerun needs fresh output directories; failures and partial results are kept.

Only WAV paths enter ASR. Context, args, action_id and reference text are never model prompts. `spoken_text_text_only.py` transforms only the original `text`; the resulting `text` is both TTS input and ASR reference. Ordinary WER/CER normalization stays unchanged; numerical canonicalization is a separate, limited diagnostic.

The generic `scripts/run_pipeline.py --config configs/pipeline-pilot.json` is an optional four-command smoke test. It performs preflight internally and runs ordinary evaluation, but **does not implement the common pre-ASR length gate or numerical evaluation**. It is not the reproduction command for the completed Zhanna experiment. The CLI has no `--preflight` option; the guide shows how to call the read-only `preflight()` function.

## Files and results

| Location | Purpose |
|---|---|
| `scripts/` | synthesis, audio audit, ASR adapters, independent evaluators and input validation |
| `configs/pipeline-zhanna-text-only-v1.0.1.json` | measured experiment's pinned configuration; existing local output is not overwritten |
| `examples/zhanna-v1/samples.jsonl`, `source.json` | exact 163-row source snapshot and source SHA256 |
| `examples/zhanna-v1/text-only-v1.0.1/` | derived texts, original→spoken mapping, rules and pronunciation flags |
| `examples/zhanna-v1/experiment-text-only-v1.0.1/ready_samples.jsonl` | fixed 151-command selection |
| `docs/reports/zhanna-text-only-v1.0.1/artifacts/` | published text metadata, predictions, timings, metrics and provenance |
| `outputs/pipeline-zhanna-text-only-v1.0.1/dataset/` | local 1359 WAV and audio audit |
| `outputs/asr-zhanna-text-only-v1.0.1-full-cached/` | completed local ASR/evaluation after two HF HTTP 429 failures; `RESULTS.md`, `analysis/`, logs |

Original local freeze files and recovery helpers include references to historical, ignored outputs; they are audit records, not portable fresh-clone launchers. Their hashes are not refreshed to accommodate publication.

The frozen local `text-only-v1.0.1/README.md` describes its pre-ASR preparation stage. Its old status statements are superseded by [STATUS.md](examples/zhanna-v1/text-only-v1.0.1/STATUS.md) and the published report; that historical README is not part of the proposed minimal publication.

## Historical and optional experiments

[Archived pilot notes](docs/experiments/legacy-asr-pilots.md) preserve the previous README, including old machine paths and four-clip results. These instructions are historical, not the current setup. `ru-spoken-*`, text-only 1.0.0, filename spelling experiments and fragment-retention probes are not the current text-only pronunciation policy. Historical implementations remain for regression tests and the numeric helper dependency; selected historical fixtures are test data, not the main experiment.

[Data contracts](docs/contracts.md) remain unchanged. `.venv/`, `.venv-gigaam/`, `.cache/`, `outputs/` and Python caches are excluded from publication. No pretrained weights or WAV are required to rerun the published metric calculations.
