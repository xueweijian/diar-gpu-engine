# M3 — CUDA Backend Plan (three-GPU family: P100 / V100 / T4)

> **Revision 2026-09-19 (v2)**: pivoted from Kaggle-P100-only to a
> three-GPU family plan. The user's production hardware is **one P100
> and one V100**; Kaggle retired its P100 stock on 2026-09-15 (notice
> 735239), so **Kaggle T4x2 becomes the development/regression
> platform** while the user's own P100/V100 are the optimization and
> acceptance targets. Supersedes M3-T4-CUDA-PLAN.md and the original
> P100-only draft. See M3-P100-RETIRED-T4-PIVOT.md for the timeline.

Status: Step 1 (profile) CLOSED, Step 2 (backend skeleton) landed in
d4fe425. Step 3 (first CUDA operator) not started. This revision
contains zero implementation — planning only, per instruction.

## 0. Hardware fleet and roles

| GPU | Die | CC | FP32 | FP16 CUDA-core | Mem BW | Mem | Role |
|---|---|---|---|---|---|---|---|
| **P100** | Pascal GP100 | sm_60 | 9.3 TF | 18.7 TF (2:1) | 732 GB/s | 16 GB HBM2 | user production **#1**; **feature floor** |
| **V100** | Volta GV100 | sm_70 | 15.7 TF | 31.4 TF (2:1) | ~900 GB/s | 16/32 GB HBM2 | user production **#2**; perf ceiling reference |
| **T4 ×2** | Turing TU104 | sm_75 | 8.1 TF | 16.2 TF (2:1) | 300 GB/s | 16 GB GDDR6 | Kaggle dev / regression / CI |

Design invariant: **one codebase, one fatbin, three native SASS
targets.** No per-GPU forks of engine logic. Per-GPU differences are
allowed only inside kernel/launch-parameter selection tables keyed by
compute capability, chosen once at init and frozen for the session
(determinism discipline, M1 v11 lesson).

Fleet-level corollaries:

- The feature floor is sm_60. Anything Volta+ (independent thread
  scheduling, wmma, L1 improvements behind new intrinsics) or Turing+
  (tensor-core mma, unified shared memory) is **banned from the
  mainline**. This is enforced by compile gate, not convention (§2).
- All three cards do FP16 at 2:1 on CUDA cores and all three run
  cuBLAS `GemmEx` with 16F inputs / 32F compute — that is the **only**
  fp16 route that is simultaneously three-card portable and
  parity-preserving (§4).
- The workload is bandwidth-bound (arithmetic intensity of the
  260×512×2048 GEMM family ≈ O(10) FLOP/byte). HBM2 on the user's two
  cards (732/900 GB/s) is the structural advantage; T4's 300 GB/s
  GDDR6 makes it the *pessimistic* dev platform — if a design is fast
  on T4 it is structurally sound, and it gets faster on user silicon.

## 1. Non-negotiables (ADR-0002, extended to the family)

1. **sm_60 feature floor** — every .cu file must compile clean under
   `-arch=sm_60`; CI/kernel enforces it (§2). P100 defines what the
   engine may use; V100/T4 merely run it faster.
2. **No tensor cores in the mainline.** P100 has none, and fp16
   accumulate violates fp32 parity. V100/T4 tensor-core experiments
   belong in a side branch, off by default, correctness-only claims.
3. **FP32 compute is the parity anchor.** FP16 *storage* (weights in
   half precision, GEMM reads 16F, accumulates 32F) is an opt-in
   bandwidth optimization, enabled only if measured error passes the
   K6 tolerance tiers on all four fixtures.
4. **PTX forward compatibility** — embed `compute_60` PTX so any
   future sm_60+ card (A100/L4/RTX…) JITs without a rebuild.
5. Numerical truth is the CPU engine (M2-green); gates are card-
   independent. Perf claims are card-specific and never transferable.

## 2. Build matrix — fixed on Step 3 day one

```
nvcc -gencode arch=compute_60,code=sm_60 \   # P100 native SASS
     -gencode arch=compute_70,code=sm_70 \   # V100 native SASS
     -gencode arch=compute_75,code=sm_75 \   # T4  native SASS
     -gencode arch=compute_60,code=compute_60  # PTX floor + future JIT
```

CMake surface (when Step 3 lands): `DIAR_CUDA_ARCHS="60;70;75"`,
`DIAR_CUDA_PTX_FLOOR=60`. GitHub CI stays CPU-only; the CUDA compile
gate runs in the Kaggle kernel (nvcc there is 12.x and compiles all
three targets — proven by p100_m3_base kernels).

**Floor enforcement without owning a P100 (the key mechanism):**
besides the fatbin, the Kaggle kernel builds a second, PTX-only
binary (`-gencode arch=compute_60,code=compute_60`, no SASS) and runs
the same parity suite on the T4 via driver JIT. The JIT executes the
sm_60 instruction-selection semantics, so Pascal-era codegen
regressions are caught on Kaggle. Caveat written into the protocol:
PTX-JIT-on-T4 is a **semantic** check, not a P100 perf predictor.

Driver fallback: if the user's cards run drivers older than CUDA 12
(R525-era), rebuild with CUDA 11.8 — it still compiles sm_60/70/75 +
compute_60. Decide from `nvidia-smi` before the first on-site bench
(§6).

## 3. Port order — locked by the Stage 1 profile

Stage 1 verdict (docs/M3-STAGE1-PROFILE-VERDICT.md): conformer
92.34 % / transformer 6.85 % / stem 0.68 % — top-3 covers 99.87 %.
Therefore:

1. **Conformer chain (17 layers)** — FF up/down GEMMs, rel-pos MHA
   (bmm chain + shift + bias), conv module (depthwise k9 + pointwise).
2. **Transformer stack (18 layers)** — plain MHA + FF, trivial after
   the conformer machinery exists.
3. Everything else (FE/mel, PE table, xscale/concat, AOSC, gate,
   head, trim) **stays CPU permanently** (< 0.5 % combined). Moving
   them would add sync points for zero gain; revisit only if the
   chunk loop's host-side overhead ever shows up in the profile.

Each operator slice now carries three checks, all on Kaggle:
(a) T4 parity vs CPU engine (G-A), (b) sm_60 clean compile of the
same sources, (c) PTX-JIT parity rerun of the gate subset.

## 4. Bandwidth-first optimization doctrine

The engine is an inference server for mid-size GEMMs and pointwise
chains; on all three cards that regime is memory-bound. The doctrine,
in payoff order:

1. **cuBLAS SGEMM first** (Step 3) — measured on T4: 0.254/0.256 ms
   for the two FF GEMMs, 0.038 ms for proj. The CPU engine spends
   ~2.2 s/chunk on the same shapes; that is three orders of headroom
   before any hand kernel is justified. Hand-written GEMMs are a
   later, measured decision, not a default.
2. **Device residency of all weights** (Step 4) — GGUF loads once,
   dequantizes once, zero H2D in steady state; fifo/spkcache state
   lives in DeviceBuffers. The chunk loop must reach ~0 bytes
   round-tripped per chunk (measured, not assumed).
3. **FP16 weight storage** (Step 5, opt-in) — halves the dominant
   traffic. cuBLAS GemmEx 16F-in/32F-accumulate keeps compute fp32;
   only the storage quantization adds error (weights only, not
   activations). Gate: four-fixture K6 tiers must stay green with
   fp16 weights before it becomes default. Expected gain: ~1.5-2×
   on the GEMM family across all three cards (bandwidth-bound).
4. **Pointwise fusion** — bias+SiLU, GLU halves, xscale+concat+PE
   slice: fuse into GEMM epilogues where cuBLAS allows (cublasLt
   epilogue, availability per-arch to be verified) or into single
   elementwise kernels (trivially portable). Fusions are ranked by
   profile share, and each must show a bandwidth win, not a FLOP win.
5. **CUDA Graphs** for the fixed streaming geometry after warm-up;
   tail chunk falls back to the eager path (same route-decision
   pattern as tailfix). Graphs kill launch overhead, which matters
   at 20-30 ms/chunk scale only if launch count is large — measure
   first, adopt second.

Per-GPU tuning tables: cuBLAS heuristics differ per arch, so operator
benches run per card at acceptance; selected algos are recorded in
the bench JSON (fingerprint) and frozen for the session.

## 5. Acceptance gates

| Gate | Content | Anchor |
|---|---|---|
| G-A | per-operator CPU-vs-GPU max_abs ≤ 1e-5 (fp32) | CPU engine (M2-green) |
| G-B | four-fixture timeline: K6 tiers verbatim (hard v12, advisory v13-mid) | q8 fixtures |
| G-C | determinism: same-input rerun bit-identical within a session | K6-style |
| G-D | perf per card: RTF vs upstream `sortformer_p100` baseline + **ordering sanity V100 ≥ P100 ≥ T4** (violation = per-card bug, e.g. wrong SASS target picked) | per-card bench JSON |
| **G-E (new)** | **cross-card numeric agreement**: same input on all three cards, probs max_abs ≤ 1e-6 (fp32 route) | three-way diff |

G-E is new in v2 and is the strongest implementation-bug catcher in
the set: three independent driver/cuBLAS stacks agreeing to 1e-6
makes a silent kernel bug vanishingly unlikely. Run at acceptance
(two rows from the user's cards, one from Kaggle).

Autotune hygiene (M1/M2 lessons) unchanged: pinned workspace + env
for gate runs; report pinned and free-running numbers; compare gates
on pinned only.

## 5b. Official baseline (measured 2026-09-19, locked)

Kernel `weijianxue/diar-official-bench` v4, archived at
`shared/diar-gpu-engine/m3-official-bench/official_bench_t4.json`:

- NeMo PyTorch eager, fp32, streaming (the only official form — the model
  has no offline `.transcribe`), torch 2.10.0+cu128, T4 (cc 7.5, 40 SM).
- short 56.86 s / 36 chunks: **45.5 ms/chunk** (wall 1.65 s, RTF 0.029)
- mid 357.29 s / 224 chunks: **46.1 ms/chunk** (p50 45.5, wall 10.37 s,
  RTF 0.029; 3-run variance <1%, warmup excluded)
- Estimated composition: ~12-16 ms PyTorch op-dispatch overhead per chunk
  (~700 eager ops) on top of ~30 ms of actual GPU compute.

**G-D extension (the M3 bar):** the CUDA engine must beat the official
baseline on the same card class — on Kaggle T4: **≤ 25 ms/chunk mean
(≥1.8x)** for the mid feed to declare the port successful; stretch goal
≤ 20 ms/chunk with fp16 storage enabled. The python-dispatch overhead we
eliminate (~30%) is the floor of the win; fusions, residency and fp16
storage are the upside. On user P100/V100 the official number is not
directly measurable on Kaggle anymore (P100 retired) — the acceptance
protocol is our engine vs the T4-official RTF scaled by card class, plus
absolute ms/chunk targets recorded at first bench.

## 6. Acceptance and bench protocol on user hardware

- Deliverable: one fatbin + `diar-bench` harness. The user runs
  `diar-bench --fixture <four fixtures> --json` natively on the P100
  and the V100 (zero Kaggle involvement).
- Bench JSON fingerprint: device name, CC, driver, CUDA/cuBLAS
  version, which SASS target actually executed, per-op p50/p95/p99,
  bytes H2D/D2H, end-to-end RTF per fixture.
- Pre-bench one-time check: `nvidia-smi` on both cards → confirm
  driver ≥ CUDA-12 era, else ship the CUDA 11.8 build (§2).
- Perf expectation to beat: T4 numbers × (bandwidth ratio ≈ 2.4×)
  as the P100 rough guide; V100 above that. These are planning
  ratios, not promises — the bench JSON is the truth.

## 7. Explicitly out of scope (M3)

- Multi-GPU (Kaggle's T4x2 included): user production is single-card;
  a dual-T4 pipeline solves no user problem. Revisit only if
  production goes multi-card.
- Training, ASR, EEND-TA, batching of concurrent streams (M5+).
- Tensor-core paths in the mainline (§1.2).
- Rewriting operators the profile already exonerated (§3.3).

## 8. Risks

| Risk | Mitigation |
|---|---|
| User driver too old for CUDA 12 PTX | CUDA 11.8 fallback build (all three archs supported); nvidia-smi check first (§6) |
| PTX-JIT-on-T4 accepted but P100 native diverges | can't happen for semantics (same ISA floor); for perf we never transfer claims — P100 bench is the only P100 truth |
| cuBLAS version drift across machines | determinism gates are per-session; cross-machine is G-E parity only, never bit-identical claims |
| Kaggle T4 quota (30 h/week GPU) | CPU kernels remain the workhorse for logic; GPU sessions only when an operator slice is ready (current practice) |
| Kaggle image drift breaks nvcc | versions pinned+logged per ADR-0002; CPU CI matrix stays the fallback truth |
| Scope creep into exotic per-card kernels | §0 invariant: one codebase; per-GPU deltas confined to selection tables |

## 9. Step map (delta vs previous plan)

| Step | Content | Status |
|---|---|---|
| 1 | profile-first | **CLOSED** 2026-09-19 (conformer 92.34 %) |
| 2 | backend skeleton (backend.hpp, DIAR_WITH_CUDA off) | **landed** d4fe425 |
| 3 | build matrix (§2) + conformer FF GEMM slice via cuBLAS + G-A on T4 + sm_60 gate + PTX-JIT rerun | next, not started |
| 4 | device-resident state (fifo/spkcache/weights) | unchanged |
| 5 | fp16 storage opt-in + fusions + graphs, each profile-gated | unchanged + §4 doctrine |
| 6 | acceptance: G-A..G-E, user-hardware bench protocol (§6) | extended by G-E + §6 |
