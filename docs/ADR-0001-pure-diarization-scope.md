# ADR-0001: Pure speaker diarization scope

- Status: accepted
- Date: 2026-09-13

## Decision

This repository implements only speaker diarization: estimating speaker activity over time and emitting speaker-labeled segments. ASR/transcription is explicitly outside the core engine.

## Consequences

- No ASR model, tokenizer, vocabulary, WER, or transcription dependency is allowed in the core build.
- NeMo-Speech.cpp is used only as a reference for its Sortformer diarization component and GGUF/C++ runtime patterns.
- ASR integration may be a future separate adapter, but it cannot alter the core API or benchmark.
- All model cards, tests, benchmarks, and CI names use `diarization` or `speaker activity`, not generic `speech-to-text`.
