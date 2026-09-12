# Contributing

This is a pure speaker diarization engine. A contribution must preserve that
boundary: no ASR model, transcription output, tokenizer, or WER dependency in
the core runtime.

## Before coding

1. Search current papers and open implementations for algorithmic work.
2. Write the intended accuracy and performance contract.
3. Add or update a small deterministic test fixture.
4. State whether the change affects frontend, neural core, streaming state, or postprocessing.

## Before opening a PR

```sh
./scripts/run_local_tests.sh
```

For model/runtime changes, attach a JSONL benchmark result with commit,
model/checkpoint hash, environment, precision, timing boundary, and parity or
DER results. Large weights and generated benchmark outputs do not belong in the
repository.

## Optimization rule

Keep a readable reference implementation. A CUDA or SIMD optimization is not
accepted because it is faster alone; it must pass numerical parity and accuracy
gates first. If a result is faster but changes quality, document it as an
explicit tradeoff rather than silently changing the default.
