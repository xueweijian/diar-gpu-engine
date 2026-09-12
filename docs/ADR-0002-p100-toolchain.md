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
