# P100 compatibility probe

This is the next short experiment after the initial smoke test.

It tests two independent paths:

1. Native CUDA `nvcc -arch=sm_60` kernel, which is the path the future pure
   diarization C++ engine will use.
2. PyTorch 2.10.0 from the CUDA 12.6 wheel index, because Kaggle's default
   cu128 wheel omits Pascal kernels.

No model weights, ASR packages, or audio datasets are downloaded.
