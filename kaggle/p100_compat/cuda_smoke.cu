#include <cstdio>

__global__ void smoke_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 1.25f + 2.0f;
}

int main() {
    cudaDeviceProp property = {};
    if (cudaGetDeviceProperties(&property, 0) != cudaSuccess) return 1;
    std::printf("device=%s cc=%d.%d\n", property.name, property.major, property.minor);
    constexpr int n = 1024;
    float host[n], out[n];
    for (int i = 0; i < n; ++i) host[i] = static_cast<float>(i);
    float *device_x = nullptr, *device_y = nullptr;
    if (cudaMalloc(&device_x, sizeof(host)) != cudaSuccess) return 2;
    if (cudaMalloc(&device_y, sizeof(out)) != cudaSuccess) return 3;
    cudaMemcpy(device_x, host, sizeof(host), cudaMemcpyHostToDevice);
    smoke_kernel<<<4, 256>>>(device_x, device_y, n);
    if (cudaGetLastError() != cudaSuccess) return 4;
    cudaMemcpy(out, device_y, sizeof(out), cudaMemcpyDeviceToHost);
    cudaFree(device_x);
    cudaFree(device_y);
    double sum = 0.0;
    for (float value : out) sum += value;
    std::printf("sum=%.1f\n", sum);
    return 0;
}
