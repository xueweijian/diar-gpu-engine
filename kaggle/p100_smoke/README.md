# Kaggle P100 smoke test

This kernel only checks the remote GPU/toolchain and performs a tiny CUDA
operation. It intentionally does not download model weights.

Run locally:

```sh
kaggle kernels push -p kaggle/p100_smoke --accelerator NvidiaTeslaP100
kaggle kernels status weijianxue/diar-gpu-engine-p100-smoke
kaggle kernels output weijianxue/diar-gpu-engine-p100-smoke -p /tmp/diar-p100-output
```

Important: recent Kaggle default PyTorch images may expose a P100 while their
CUDA 12.8 wheel lacks Pascal `sm_60` kernels. A failed torch operation is an
environment result, not proof that the engine or GPU is broken. Record the
failure and use a Pascal-compatible wheel/toolchain or native C++ CUDA build.
