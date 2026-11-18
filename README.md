# Voice Models

Research project for converting Russian voice commands into structured actions
using descriptions of available actions and application context. The planned
pipeline uses ASR for transcription and a BERT-based model for action selection
and argument extraction.

## Status

Use of the synthetic text generator is temporarily frozen because generated
commands have insufficient linguistic quality and can disagree with their
action and argument labels. The code and experimental dataset in `data/v1/`
are retained for reproducibility and error analysis.

Dataset version `data/v1.1/` uses Russian examples from MASSIVE and an action
catalog from MASSIVE-Agents.

## Text dataset adapter

Imports Russian examples from MASSIVE and action definitions from MASSIVE-Agents.
[Adapter README](dataset-generator/massive/README.md)

## Synthetic text generator

Generates synthetic Russian voice commands with action labels and arguments.
[Generator README](dataset-generator/text/README.md)
