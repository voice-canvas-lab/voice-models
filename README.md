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

