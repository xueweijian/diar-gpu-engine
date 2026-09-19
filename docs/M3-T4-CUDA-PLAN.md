# M3 — T4 CUDA Backend Plan (was P100; see M3-P100-RETIRED-T4-PIVOT.md)

> **2026-09-19**: Kaggle retired the P100 on 2026-09-15 (official notice
> 735239). Target accelerator is now **T4x2 (sm_75)**; all porting work
> below is unchanged, only the optimization target moves. fp16 route is
> re-evaluated (tensor core on T4), first pass stays pure fp32.

Status: DRAFT — unlock condition is M2 closure (K5 v4 patched-engine green
+ K6 advisory judgement). Zero implementation has started. This document
exists so M3 can open the same hour M2 closes.

## 0. Assets already in place

| Asset | Where | Note |
|---|---|---|
| CPU parity engine | `src/` (FE/stem/17 conformer/18 transformer/AOSC/gate/engine) | M2 truth anchor — every GPU step diffs against it |
| q8 GGUF loader | `src/gguf.cpp` | dequant-to-fp32 on load; feeds both backends |
| Four-fixture gate | K6 (v12-short/mid-full hard, v13-mid advisory) | reusable verbatim as the GPU acceptance gate |
| NeMo npz anchors | `shared/diar-gpu-engine/m2-ref/` | K5-style open/closed-loop checks |
| Synthetic bench | `tools/bench_engine_synth.py` | d128→566ms, d256→1878ms, ×3.3 ≈ d² scaling; extrapolates real d=512 config to ~30–40 s/chunk CPU |
| P100 probes | `kaggle/p100_smoke`, `kaggle/p100_compat` | nvcc `-arch=sm_60` native kernel compiles and runs; torch needs cu126 wheel (cu128 omits Pascal) |
| P100 ggml baseline | `kaggle/sortformer_p100` | upstream binary RTF numbers to beat |
| ADR-0002 | `docs/` | P100-first, FP32 & FP16-storage first-class, no Tensor Cores, no BF16/FP8 assumptions |

## 1. Scope and non-goals

Goal: same DiarEngine, CUDA execution of the neural core on P100, bit-level
row geometry, numeric parity vs the CPU engine, measurable RTF.

Non-goals (M3): training, multi-GPU, non-Pascal targets (any result from
another GPU is correctness-only, not a perf claim — ADR-0002), fp16
accumulation, ASR anything.

## 2. Step 1 — profile before porting (1 kernel + local work)

- Kaggle CPU kernel: per-stage timing of the existing engine on the real
  d=512 config (bench_engine_synth already gives the shape; add per-stage
  taps behind a compile flag, no behavior change).
- Expected split (to verify): conformer GEMMs (17 layers × {qkv, pw, ff})
  + transformer (18) dominate; FE/mel and AOSC bookkeeping minor.
- Output: `m3_stage1_profile.json` + decision note on which 3–5 operators
  cover ≥80% of time. Do not port anything before this lands.

## 2b. Step 1 verdict (2026-09-19, docs/M3-STAGE1-PROFILE-VERDICT.md)

conformer 92.34 % / transformer 6.85 % / stem 0.68 % — top3 99.87 %.
Port order settled: the 17-layer conformer chain first; everything else
stays CPU-side. Step 2 (backend skeleton) landed in d4fe425.

## 3. Step 2 — backend skeleton (local + CI-compiled, CPU-run)

- `include/diar/backend.hpp`: opaque DeviceBuffer, Stream, and an
  operator interface mirroring the ~11 `nn::` primitives. CPU keeps the
  existing implementation; CUDA implementation returns
  `unsupported` at first.
- CMake `DIAR_WITH_CUDA` option (OFF by default; GitHub Actions stays
  CPU-only). Alpine/MSVC must still configure+build clean with OFF.
- Host-side geometry identical: the engine calls the same shapes; the
  backend decides where memory lives. No engine-logic changes allowed in
  this step (tests pin that the CPU path is untouched).

## 4. Step 3 — operator migration, one PR-sized chunk each

Order (by expected time share, re-ranked after Step 1):

1. `linear` (GEMM) via **cuBLAS** FP32 — decision: buy speed and
   correctness first, hand kernels later. cuBLAS SGEMM on Pascal is
   strong; ggml-cuda remains the reference for what is achievable.
2. `layernorm`, `softmax`, activation fusions (bandwidth-bound,
   straightforward kernels, good first hand-CUDA targets).
3. rel-pos attention path (shift + bias add + bmm chain) — keep FP32
   accumulation end-to-end (P100-safe, ROADMAP M3.4).
4. conv stem / depthwise / subsampling.
5. head + gate stay CPU-side initially (tiny), move only if profiling
   says so.

Each chunk: local CUDA-less unit tests via a golden-binary fixture
(pinned outputs from the P100 kernel, compared CPU-vs-GPU in the Kaggle
kernel), plus the four-fixture gate re-run per milestone (Stage 3.6).

## 5. Step 4 — device-resident state

- Move `fifo`/`spkcache`/mean-silence buffers to DeviceBuffer; chunk
  loop stays on host, payloads never round-trip except for taps.
- AOSC taps (K5 G3-style) become opt-in copies for parity runs only.
- Measure: bytes transferred per chunk before/after must drop to ~0 in
  steady state.

## 6. Step 5 — P100-specific optimization (only after parity green)

- FP16 storage + FP32 accumulation for weights (halves bandwidth; P100
  FP16 arithmetic throughput 2× FP32 — decide per operator from the
  Step 1 profile).
- Fuse: depthwise conv + pointwise + residual; mel/filterbank if FE
  shows up in the profile.
- CUDA Graph replay for the stable streaming geometry (fixed chunk
  shape after warm-up; tail chunk falls back to the eager path — same
  pattern as the tailfix route decision).
- Metrics protocol (ROADMAP M3.7): launches, occupancy, device mem,
  H2D/D2H bytes, p50/p95/p99 per chunk and end-to-end RTF.

## 7. Gates

| Gate | Content | Anchor |
|---|---|---|
| G-A | per-operator CPU-vs-GPU max_abs ≤ 1e-5 (fp32) | CPU engine (M2-green) |
| G-B | four-fixture timeline: reuse K6 tiers (hard v12, advisory v13-mid) | q8 fixtures |
| G-C | determinism: same-input GPU reruns bit-identical within a session | K6-style |
| G-D | perf: RTF vs `sortformer_p100` upstream baseline; no regression clause — M5 quality gate keeps frame agreement |

Autotune hygiene learned in M1/M2: pin CUBLAS workspace + deterministic
env for gate runs; report both pinned and free-running numbers, compare
gates on pinned only.

## 8. Risks

- cuBLAS nondeterminism across processes (v11 lesson): gates must pin
  or re-judge like K6 advisory — never chase a moving anchor.
- Kaggle image drift breaking the CUDA toolchain — pin and log versions
  per ADR-0002, keep the CPU CI matrix green as the fallback truth.
- Scope creep into M4 (offline extreme throughput): EEND-TA stays out.
