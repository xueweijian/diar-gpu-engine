# ADR-0002: P100-first validation

- Status: accepted
- Date: 2026-09-13

## Decision

The primary target GPU is Tesla P100 16GB, Pascal `sm_60`. GitHub Actions proves portable C++ correctness; Kaggle P100 proves CUDA behavior and performance.

## Constraints

- The P100 has no Tensor Cores.
- Recent CUDA/PyTorch binary combinations may omit Pascal kernels.
- A current default Kaggle PyTorch image can report a visible P100 while failing the first custom CUDA operation.
- Model weights and large toolchains must not be stored on the phone.

## Policy

- Pin and log CUDA/toolkit/driver versions for every GPU run.
- Prefer native C++/CUDA or a Pascal-compatible runtime over assuming the newest PyTorch wheel works.
- Use `sm_60` explicitly when compiling kernels.
- Treat FP32 and FP16-storage + FP32-accumulation as first-class P100 paths; do not assume BF16/FP8/INT8 acceleration.
- Any result from a non-P100 GPU is useful for correctness but is not a P100 performance claim.

---

## Revision 2026-09-19 — three-GPU family (ADR-0002 still stands, scope widened)

Kaggle retired its P100 stock on 2026-09-15 (notice 735239). The user's
production hardware is a **P100 and a V100**; Kaggle T4x2 is now the
dev/regression platform. Everything this ADR decided holds and extends:

- P100 remains the **toolchain and feature baseline** (sm_60 floor) — now
  doubly so, because the user owns the card the floor is named after.
- FP32 & FP16-storage remain first-class; FP16 route is cuBLAS GemmEx
  16F-in/32F-compute (portable across sm_60/70/75, parity-preserving).
- "No Tensor Cores, no BF16/FP8" is now a **family constraint**, not just
  a Pascal one: the P100 in the fleet has no tensor cores, so the
  mainline cannot depend on them. V100/T4 tensor-core work stays in a
  side branch, correctness-only.
- New: one fatbin with sm_60/sm_70/sm_75 SASS + compute_60 PTX (forward
  compat for future cards). See M3-CUDA-PLAN.md §2.
- Non-Pascal results upgraded from "correctness-only" to **first-class
  perf targets on the user's own silicon** (V100 ≥ P100 ≥ T4 ordering
  sanity is part of gate G-D).
