# diar-gpu-engine

Pure speaker diarization inference engine for NVIDIA GPUs.

> **Scope:** this project answers only **who spoke when**. It is not an ASR project. No transcription model, tokenizer, WER benchmark, or ASR dependency belongs in the core engine.

The first target is a reproducible, high-performance implementation of **NVIDIA Streaming Sortformer v2** on Tesla P100 (Pascal, `sm_60`). The second target is an offline extreme-throughput branch based on **EEND-TA / EEND-M2F**.

## Why this project exists

Existing diarization stacks often combine VAD, speaker embeddings, clustering, Python orchestration, and redundant sliding windows. That is useful for generality but leaves substantial room for a small, purpose-built GPU runtime:

- fixed tensor shapes and preallocated buffers;
- fused feature extraction and neural operators;
- streaming state kept on device;
- CUDA Graph replay for repeated shapes;
- deterministic parity against a trusted reference;
- benchmark logs that separate frontend, neural core, postprocessing, and end-to-end wall time.

## Current status

**M0.1 — pure-diarization contract and reference postprocessing scaffold.**

The repository currently contains a dependency-free C++17 reference core, deterministic unit tests, benchmark result schema, CI, and a Kaggle P100 smoke harness. Model weights are intentionally not stored here.

## Roadmap

1. **M0 — contracts and test fixtures**: frame matrix, segments, postprocessing, golden fixtures, reproducible logs.
2. **M1 — Sortformer v2 parity**: Python/NeMo reference dumps versus NeMo-Speech.cpp/GGUF and this runtime.
3. **M2 — CUDA engine**: fbank, subsampling, Fast-Conformer, relative-position attention, Transformer head, AOSC/FIFO state.
4. **M3 — P100 optimization**: `sm_60`, FP32/FP16 storage experiments, CUDA Graph, kernel fusion, memory/launch profiling.
5. **M4 — offline extreme RTF**: EEND-TA first, EEND-M2F second, with a separate accuracy/latency contract.
6. **M5 — optional v2.1 parity**: only after v2 is stable; no weight download on the phone.

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/TESTING.md](docs/TESTING.md).

## Development rules

- Tests before optimization; every new operator gets a scalar/reference test.
- Logs before claims; every benchmark records commit, compiler, CUDA driver/toolkit, GPU, precision, shape, warmup, and timing percentiles.
- Search first; algorithmic changes must cite the relevant paper, implementation, or benchmark.
- Do not download model weights or large dependencies to the phone. Use GitHub Actions for normal CI and Kaggle CLI for P100 experiments.
- Never compare a core-only RTF with an end-to-end RTF without labeling the boundary.
- Preserve a slow, readable reference path. Fast kernels are accepted only after parity tests pass.

## Local reference tests

The phone does not need CMake for the first scaffold:

```sh
./scripts/run_local_tests.sh
```

On a normal development host:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## GPU validation

The Kaggle harness is under [`kaggle/p100_smoke`](kaggle/p100_smoke). It is deliberately small and does not download model weights. Later GPU jobs will download/cache weights inside the ephemeral Kaggle runtime and emit only compact JSON benchmark artifacts.

Kaggle's current default PyTorch images may report a P100 but lack Pascal kernels in recent CUDA 12.8 wheels. The project therefore treats framework compatibility as a logged experimental variable and will prefer a native C++/CUDA path or a Pascal-compatible toolchain.

## Licensing

The engine code is MIT licensed. NVIDIA model weights and NeMo-Speech.cpp remain under their respective licenses; this repository does not redistribute them.
