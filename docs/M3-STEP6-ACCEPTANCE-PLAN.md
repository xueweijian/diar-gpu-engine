# M3 Step 6 — engine-level integration + diar-bench acceptance

Status: in flight 2026-09-19. Scope: wire the CUDA encoder chain into
`DiarEngine` (the missing production shape — steps 3-5 proved the operators
in harness geometry; nothing has run the real q8 GGUF through the GPU yet),
then ship the §6 deliverable (`diar-bench` + single fatbin) and run the
acceptance gates as far as Kaggle T4 allows.

## 6a. EncoderRoute hook (single source of truth)

- `EncoderRoute` abstract interface in `sortformer.hpp` (no CUDA types —
  CPU builds link it inert): `encoder_forward(x[L,D], pe[2L-1,D],
  px[L,X], L, cfg)` replaces steps 5-7 (conformer chain + encoder_proj +
  transformer chain) of `sortformer_run_chunk` when it returns true.
  `false` = CPU fallback (L beyond the route's sized budget).
- `TapSink` forces the CPU chain (K5 per-layer tap semantics are a CPU
  contract; a routed run never fires layer taps).
- `DiarEngine::set_encoder_route(EncoderRoute*)` threads it through
  `run_one_chunk`. `diarize_offline` (whole-file, L up to 4467) stays CPU
  — the v12-mid-offline-full fixture runs that path by design.
- Route=nullptr (all existing callers) must stay bit-identical: pinned by
  `tests/test_encoder_route.cpp` (baseline equality + mock-route wiring +
  taps-override + config echo).

## 6b. CudaEncoderRoute (DIAR_WITH_CUDA TU)

- `src/encoder_cuda.cpp`: owns GpuArena + cublas handle + stream; uploads
  conformer×17, transformer×18 and encoder_proj from a real
  `SortformerWeights` at construction (fp32 Sgemm or fp16-storage GemmEx
  route — the same template bodies step 4/5 gated). Per chunk: one H2D of
  x+pe, device chain, one D2H of px. Single stream, fixed kernel order —
  determinism by construction (G-C rerun must be bit-identical).
- Sized from `max_l` (ctor arg; streaming geometry ⇒ L ≤ ~266). L > max_l
  returns false ⇒ CPU fallback, never a silent wrong answer. Counts
  calls/fallbacks for the bench JSON.

## 6c. diar-bench (the §6 deliverable)

`tools/diar_bench_main.cpp` — `diar-bench --weights <gguf|dfw1> --audio
in.f32 [--ref probs.f32] --mode streaming|offline-preset|full-offline
--route cpu|fp32|fp16 --reps N --out prefix`:

- fingerprint: device name, CC, SM count, driver/runtime, cuBLAS version,
  route, fatbin SASS set (offline fact) + which CC executed.
- per case: rows, ledger tail data, ms/chunk (wall / ledger size), RTF,
  per-stage profile (DIAR_PROFILE_STAGE build), determinism bit-identity
  across reps.
- vs-ref compare (when `--ref` given): K6 metrics (max_abs, mean_abs,
  frame_agreement, row percentiles) — same formulas as the K6 kernel so
  numbers are comparable across runs.
- outputs `prefix.bench.json` + `prefix.pregate.f32` / `prefix.postgate.f32`
  in the probdump wire format (12B `<qi>` header + payload) — the G-E
  cross-card artifacts.
- CPU selftest mode (no CUDA): tiny fixture → mock route wiring + wire
  round-trip + metrics — a Kaggle round trip never burns on harness bugs.

## 6d. Gate mapping (kernel m6_step6, T4)

| Gate | In this step | Anchor |
|---|---|---|
| G-A | operator parity (steps 3-5, continuously green; regression via CI) | CPU engine |
| G-B | 4 fixtures × routes {cpu, fp32, fp16} vs fixture probs, K6 tiers verbatim (v12 hard, v13-mid advisory) | q8 fixtures |
| G-B2 (new) | **route-vs-cpu same-engine diff**: fp32 route vs cpu route on the same fixture — the sharpest integration gate (same weights, same host state machine; only the encoder chain differs). v1 measures, v2 pins calibrated gates (sqrt-walk discipline from step 5) | own baseline |
| G-C | per case+route: rerun bit-identical | K6-style |
| G-D | wall ms/chunk + RTF per route; hard ≤25 ms, stretch ≤20 (fp16) | official 45.5 T4 |
| G-E | probs dumps emitted per route; three-way diff runs when the user's P100/V100 rows arrive (scripts/ge_cross_diff.py, ≤1e-6 fp32) | cross-card |

fp16 stays opt-in (default off) until it passes the K6 four-fixture gate
through this engine-level path — pre-agreed Step 5 semantics.

## 6e. User-hardware protocol (unchanged from plan §6)

nvidia-smi driver check → build (CUDA 11.8 fallback if pre-R525) →
`diar-bench --fixture ... --json` per card → send back bench JSON + probs
dumps → G-E three-way + G-D ordering sanity V100 ≥ P100 ≥ T4.

## Non-goals

- GPU stem / PE / concat / head (profile-exonerated, <1% — permanent CPU).
- Full-offline on GPU (L=4467 attention scratch ≈ heads·L² — not worth it
  for one fixture; CPU route answers it).
- CUDA Graphs, host/GPU pipelining: post-acceptance options only.
