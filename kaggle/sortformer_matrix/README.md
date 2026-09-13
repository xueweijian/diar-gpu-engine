# Sortformer v2 P100 measurement matrix

Consumes the published harness dataset `weijianxue/diar-gpu-engine-harness`
(mounted at `/kaggle/input/diar-gpu-engine-harness`) so the build, model
download and measurement code has a single source of truth.

Questions answered:

1. marginal compute per audio second vs fixed model-load cost (length fit);
2. whether throughput degrades on long inputs (state growth);
3. CPU fallback cost on the same binary;
4. streaming vs offline geometry cost;
5. byte-identical RTTM across repeated runs (parity prerequisite);
6. whether `--concurrency` directory batching helps on a single P100.

Outputs (small, reviewable): `sortformer_matrix_report.json`,
`benchmarks.jsonl` (schema `benchmarks/schema/benchmark-v1.schema.json`),
`run_tail.log`. Build trees and weights stay in `/tmp` and are discarded.
