// Pure diarization P100 toolchain smoke test without PyTorch.
// Compile with CUDA 12.6 and sm_60; prints GPU architecture and a checksum.
#include <cstdio>

__global__ void smoke_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * 1.25f + 2.0f;
    }
}

int main() {
    cudaDeviceProp property = {};
    if (cudaGetDeviceProperties(&property, 0) != cudaSuccess) {
        std::printf("no-cuda-device\n");
        return 1;
    }
    std::printf("device=%s cc=%d.%d\n", property.name, property.major, property.minor);
    constexpr int n = 1024;
    float host[n];
    float out[n];
    for (int i = 0; i < n; ++i) {
        host[i] = static_cast<float>(i);
    }
    float *device_x = nullptr, *device_y = nullptr;
    cudaMalloc(&device_x, sizeof(host));
    cudaMalloc(&device_y, sizeof(out));
    cudaMemcpy(device_x, host, sizeof(host), cudaMemcpyHostToDevice);
    smoke_kernel<<<4, 256>>>(device_x, device_y, n);
    cudaMemcpy(out, device_y, sizeof(out), cudaMemcpyDeviceToHost);
    cudaFree(device_x);
    cudaFree(device_y);
    double sum = 0.0;
    for (float value : out) {
        sum += value;
    }
    std::printf("sum=%.1f\n", sum);
    return 0;
}
