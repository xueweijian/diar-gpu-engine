# Testing and benchmark protocol

## Non-negotiable scope

The core accepts audio/features and returns speaker activity probabilities or speaker segments. It does not transcribe. Tests must not import Whisper, Parakeet, Nemotron, tokenizer, or ASR packages.

## Test pyramid

### L0 — pure scalar tests

- matrix shape and bounds;
- threshold hysteresis;
- overlapping speakers;
- frame-to-time conversion;
- padding, gap filling, minimum duration;
- malformed input and deterministic error messages.

### L1 — reference tensor tests

Small checked-in fixtures only. Compare every layer or operator against a simple CPU implementation. Fixtures must be small enough for GitHub and the phone.

### L2 — backend parity

For the same fixture and fixed seed:

- CPU reference vs CPU optimized;
- CPU reference vs CUDA FP32;
- CUDA FP32 vs CUDA FP16-storage;
- final frame probabilities and final segments.

Report absolute max error, mean absolute error, cosine similarity, and binary frame agreement at the configured threshold.

### L3 — real corpus accuracy

Large datasets and model weights stay off the phone. Run on Kaggle or another disposable GPU workspace. Report DER, JER, missed speech, false alarm, confusion, overlap-only DER, and speaker-count buckets.

### L4 — performance

Every performance run records:

- repository URL and commit;
- model/checkpoint ID and hash;
- dataset/fixture ID and hash;
- OS, compiler, CUDA toolkit, driver, GPU name, compute capability;
- precision, quantization, batch, chunk, left/right context, FIFO and AOSC sizes;
- warmup count and measured iteration count;
- frontend time, neural-core time, postprocess time, end-to-end wall time;
- p50/p90/p95/p99 latency, RTF, realtime factor, peak host RAM, peak VRAM;
- pass/fail against parity and accuracy gates.

## RTF conventions

Use both conventions and label them:

```text
rtf = wall_time_seconds / audio_duration_seconds
realtime_x = audio_duration_seconds / wall_time_seconds
```

Never call a log-Mel-only number an end-to-end RTF. The benchmark report must state whether it includes decoding, resampling, fbank, model load, postprocessing, and file I/O.

## Accuracy gates

Default gates for an optimization PR:

- scalar/operator tests: exact or documented floating-point tolerance;
- frame probability: `max_abs_error <= 1e-4` for FP32 reference unless the operator has a reviewed tolerance;
- binary frame agreement: `>= 99.9%` on the golden fixture;
- segment boundary drift: `<= 1 frame` unless the PR changes postprocessing intentionally;
- no DER regression beyond the PR's declared tolerance on the sentinel set.

These are initial gates, not claims about final model quality. Model-specific gates will be added after Sortformer v2 parity is established.

## Logs

Use JSON Lines for machine-readable results. Keep human summaries in Markdown. A failed benchmark is still an artifact: record the environment and the first failing layer instead of deleting the run.
