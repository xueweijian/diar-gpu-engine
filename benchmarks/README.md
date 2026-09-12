# Benchmarks

This directory contains schemas and small benchmark tooling only. Model weights,
large datasets, and raw result archives stay in Kaggle or another disposable
runtime.

Every result must identify whether it measures:

- `frontend`: audio decode/resample/fbank;
- `neural_core`: model execution only;
- `postprocess`: thresholding/AOSC/segment conversion;
- `end_to_end`: the complete pure-diarization path.

Use JSON Lines for results. The schema is
[`schema/benchmark-v1.schema.json`](schema/benchmark-v1.schema.json).
