# Roadmap

## M0 — reproducibility contract

### M0.1 (current)

- Define pure diarization data contracts.
- Implement a scalar/reference frame-probability postprocessor.
- Add deterministic unit tests for hysteresis, overlap, malformed input, and segment timing.
- Add a benchmark result schema.
- Add GitHub Actions on Linux, Windows, and macOS.
- Add a Kaggle P100 smoke job that records GPU/toolchain state without downloading weights.

### M0 exit criteria

- All unit tests pass on the three GitHub-hosted OSes.
- Sanitizer job passes on Ubuntu.
- No ASR dependency or ASR output is present in the core API.
- A benchmark log can be reproduced from a commit and a small fixture.

## M1 — Sortformer v2 numerical truth

Do not download the 471 MB/147 MB model artifacts to the phone. Use a Kaggle runtime or a disposable CI/cloud workspace.

1. Generate a tiny deterministic audio fixture and reference tensors with NeMo.
2. Store only small metadata, hashes, and reduced golden tensors in GitHub.
3. Compare:
   - fbank output;
   - subsampling output;
   - each Conformer block;
   - Transformer encoder output;
   - four speaker probabilities;
   - frame-level labels and final segments.
4. Add tolerances by precision and record them in the manifest.

## M2 — native engine

- C++ model loader and tensor arena.
- FP32 reference backend.
- CUDA backend targeting `sm_60`.
- Static-shape and streaming-shape execution.
- Device-resident FIFO/AOSC state.
- CPU and GPU outputs compared on the same fixture.

M2 CLOSED 2026-09-19: K5 v4 k5a-green (patched engine, FE tail-frame
exemption) + K6 v5 hard-gate green with v13-mid advisory (fixture-side GPU
autotune noise) + tailfix. See m2-stage3/M2-STAGE3-CLOSURE.md. CUDA-side
delivery moved to [M3-P100-CUDA-PLAN.md](M3-P100-CUDA-PLAN.md).

## M3 — P100 optimization

Optimize only after M2 parity is green:

1. Profile end-to-end and neural-core boundaries separately.
2. Fuse Mel/filterbank operations where useful.
3. Fuse depthwise convolution and pointwise/residual operations.
4. Implement relative-position attention with P100-safe FP32 accumulation.
5. Benchmark FP32, FP16 storage + FP32 accumulation, and Q8 bandwidth paths.
6. Add CUDA Graph replay for stable streaming geometry.
7. Measure kernel launches, occupancy, device memory, host-device transfers, and p50/p95/p99 latency.

## M4 — offline extreme-throughput branch

Implement EEND-TA first because its decoder is parallel and its model is small. EEND-M2F follows only if a reproducible checkpoint and reference are available.

The offline branch must not silently replace the streaming Sortformer metrics. It gets its own model card, fixtures, DER/JER table, and RTF protocol.

## M5 — quality and robustness

- v2.1 meeting-domain parity.
- language and acoustic-domain sentinel set.
- overlap-heavy sentinel set.
- 1–4 speaker scaling tests.
- long-form state-compaction tests.
- regression gate: no speed optimization may reduce frame agreement or DER beyond an explicit reviewed threshold.
