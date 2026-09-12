# M1 — Sortformer v2 parity plan

This is the next model milestone. It remains **pure diarization**: the output
is frame-level speaker activity and segments; ASR is not part of the plan.

## Reference boundaries

The first comparison will use NVIDIA's `diar_streaming_sortformer_4spk-v2` and
its published GGUF/C++ runtime path. Do not store the model on the phone.
Use a disposable Kaggle runtime or a remote build workspace.

Capture, for one deterministic short WAV fixture:

1. resampled mono waveform;
2. 128-bin Mel/filterbank output;
3. post-subsampling encoder output;
4. each NEST/Fast-Conformer block output;
5. encoder projection output;
6. each post-encoder Transformer block output;
7. four speaker probabilities;
8. streaming state snapshots at chunk boundaries;
9. final frame labels and segments.

The reference dump must include model ID, file hash, NeMo commit/version,
stream geometry, precision, sample rate, and fixture hash. Store only compact
reduced tensors or hashes in Git; keep full dumps in Kaggle artifacts.

## Acceptance gates

- fbank: max absolute error agreed before implementation;
- FP32 operator parity before any quantization;
- final probability parity and frame agreement on every chunk;
- segment boundaries within one 80 ms output frame;
- no quality comparison based only on visually similar RTTM text.

## Performance runs

Report separately:

- frontend;
- neural core;
- AOSC/FIFO state update;
- postprocessing;
- end-to-end.

Warm up CUDA Graphs before timing. Record p50/p95/p99 and VRAM. A P100
performance claim requires `sm_60` hardware; A6000/Ada numbers are only
functional/profiling references.
