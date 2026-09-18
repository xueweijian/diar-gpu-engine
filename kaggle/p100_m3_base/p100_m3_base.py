# M3 base: P100 nvcc + cuBLAS + FP16-storage baseline (no model download).
# Deliverables: nvcc/sm_60 sanity, SGEMM latency on engine GEMM shape
# families, GemmEx FP16-storage/FP32-accumulate, naive cross-check.
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/tmp/m3base")
CU = "#include <cublas_v2.h>\n#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdlib>\n#include <vector>\n\n#define CK(x) do { if ((x) != cudaSuccess) { std::printf(\"CUDA_ERR @%d\\n\", __LINE__); std::exit(2); } } while (0)\n#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::printf(\"CUBLAS_ERR %d @%d\\n\", (int)s_, __LINE__); std::exit(3); } } while (0)\n\nstruct Gemm { int m, n, k; const char* tag; };\n\n__global__ void naive_sgemm(const float* A, const float* B, const float* C,\n                            float* D, int M, int N, int K) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n    if (row >= M || col >= N) return;\n    float acc = 0.f;\n    for (int t = 0; t < K; ++t) acc += A[(size_t)row * K + t] * B[(size_t)t * N + col];\n    D[(size_t)row * N + col] = acc + C[col];\n}\n\nstatic float time_ms(cudaEvent_t a, cudaEvent_t b) {\n    float ms = 0.f;\n    CK(cudaEventElapsedTime(&ms, a, b));\n    return ms;\n}\n\nint main() {\n    cudaDeviceProp p = {};\n    CK(cudaGetDeviceProperties(&p, 0));\n    std::printf(\"device=%s cc=%d.%d sm=%d\\n\", p.name, p.major, p.minor, p.multiProcessorCount);\n    cublasHandle_t h;\n    CB(cublasCreate(&h));\n\n    const Gemm shapes[] = {\n        {260, 2048, 512, \"ff_up\"},\n        {260, 512, 2048, \"ff_down\"},\n        {260, 256, 512, \"proj\"},\n        {260, 4, 256, \"head_spk\"},\n    };\n    const int NS = (int)(sizeof(shapes) / sizeof(shapes[0]));\n    std::printf(\"%-8s %6s %6s %6s %12s %12s %12s %12s\\n\",\n                \"gemm\", \"m\", \"n\", \"k\", \"f32_cold_ms\", \"f32_hot_ms\", \"f16s_hot_ms\", \"tflops_f32\");\n    float f32_hot[8], f16_hot[8];\n    for (int i = 0; i < NS; ++i) {\n        const Gemm& g = shapes[i];\n        size_t na = (size_t)g.m * g.k, nb = (size_t)g.k * g.n, nc = (size_t)g.m * g.n;\n        std::vector<float> ha(na), hb(nb), hc(g.n, 0.5f);\n        srand48(7 + i);\n        for (auto& v : ha) v = (float)(drand48() * 2 - 1);\n        for (auto& v : hb) v = (float)(drand48() * 2 - 1);\n        float *da, *db, *dc, *dd;\n        CK(cudaMalloc(&da, na * 4)); CK(cudaMalloc(&db, nb * 4));\n        CK(cudaMalloc(&dc, nc * 4)); CK(cudaMalloc(&dd, nc * 4));\n        CK(cudaMemcpy(da, ha.data(), na * 4, cudaMemcpyHostToDevice));\n        CK(cudaMemcpy(db, hb.data(), nb * 4, cudaMemcpyHostToDevice));\n        CK(cudaMemcpy(dc, hc.data(), g.n * 4, cudaMemcpyHostToDevice));\n        __half* dh16a; __half* dh16b;\n        CK(cudaMalloc(&dh16a, na * 2)); CK(cudaMalloc(&dh16b, nb * 2));\n        std::vector<__half> h16a(na), h16b(nb);\n        for (size_t j = 0; j < na; ++j) h16a[j] = __float2half(ha[j]);\n        for (size_t j = 0; j < nb; ++j) h16b[j] = __float2half(hb[j]);\n        CK(cudaMemcpy(dh16a, h16a.data(), na * 2, cudaMemcpyHostToDevice));\n        CK(cudaMemcpy(dh16b, h16b.data(), nb * 2, cudaMemcpyHostToDevice));\n        const float one = 1.f, zero = 0.f;\n        cudaEvent_t e0, e1;\n        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));\n\n        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, g.n, g.m, g.k,\n                       &one, db, g.n, da, g.k, &zero, dd, g.n));\n        CK(cudaEventRecord(e0));\n        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, g.n, g.m, g.k,\n                       &one, db, g.n, da, g.k, &zero, dd, g.n));\n        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));\n        float cold = time_ms(e0, e1);\n\n        const int iters = 200;\n        CK(cudaEventRecord(e0));\n        for (int it = 0; it < iters; ++it)\n            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, g.n, g.m, g.k,\n                           &one, db, g.n, da, g.k, &zero, dd, g.n));\n        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));\n        float hot = time_ms(e0, e1) / iters;\n\n        float f16ms = -1.f;\n        cublasStatus_t gs = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,\n            g.n, g.m, g.k, &one, dh16b, CUDA_R_16F, g.n, dh16a,\n            CUDA_R_16F, g.k, &zero, dd, CUDA_R_32F, g.n,\n            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);\n        if (gs == CUBLAS_STATUS_SUCCESS) {\n            CK(cudaEventRecord(e0));\n            for (int it = 0; it < iters; ++it)\n                CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,\n                    g.n, g.m, g.k, &one, dh16b, CUDA_R_16F, g.n, dh16a,\n                    CUDA_R_16F, g.k, &zero, dd, CUDA_R_32F, g.n,\n                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));\n            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));\n            f16ms = time_ms(e0, e1) / iters;\n        } else {\n            std::printf(\"gemm %s: GemmEx unsupported (%d)\\n\", g.tag, (int)gs);\n        }\n        double tf = 2.0 * g.m * g.n * g.k / (hot * 1e-3) / 1e12;\n        std::printf(\"%-8s %6d %6d %6d %12.3f %12.4f %12.4f %12.5f\\n\",\n                    g.tag, g.m, g.n, g.k, cold, hot, f16ms, tf);\n        f32_hot[i] = hot; f16_hot[i] = f16ms;\n\n        if (i == 0) {\n            // col-major cuBLAS dd(n,m) == row-major D(m,n) transpose; verify\n            // by comparing against a host-side reference of the same call.\n            std::vector<float> cublas_row(nc);\n            CK(cudaMemcpy(cublas_row.data(), dd, nc * 4, cudaMemcpyDeviceToHost));\n            std::vector<float> cublas_rowmajor(nc);\n            for (int r = 0; r < g.m; ++r)\n                for (int c = 0; c < g.n; ++c)\n                    cublas_rowmajor[(size_t)r * g.n + c] = cublas_row[(size_t)c * g.m + r];\n            dim3 blk(16, 16);\n            dim3 grd((g.n + 15) / 16, (g.m + 15) / 16);\n            naive_sgemm<<<grd, blk>>>(da, db, dc, dd, g.m, g.n, g.k);\n            CK(cudaDeviceSynchronize());\n            std::vector<float> naive(nc);\n            CK(cudaMemcpy(naive.data(), dd, nc * 4, cudaMemcpyDeviceToHost));\n            double max_abs = 0; size_t nbad = 0;\n            for (size_t j = 0; j < nc; ++j) {\n                double d = std::fabs(naive[j] - cublas_rowmajor[j]);\n                if (d > max_abs) max_abs = d;\n                if (d > 1e-2) ++nbad;\n            }\n            std::printf(\"naive_vs_cublas max_abs=%.3e nbad>1e-2=%zu/%zu\\n\", max_abs, nbad, nc);\n        }\n        cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);\n        cudaFree(dh16a); cudaFree(dh16b);\n        cudaEventDestroy(e0); cudaEventDestroy(e1);\n    }\n    std::printf(\"JSON {\");\n    for (int i = 0; i < NS; ++i)\n        std::printf(\"%s\\\"%s\\\":{\\\"f32_hot_ms\\\":%.5f,\\\"f16_hot_ms\\\":%.5f}\",\n                    i ? \",\" : \"\", shapes[i].tag, f32_hot[i], f16_hot[i]);\n    std::printf(\"}\\n\");\n    cublasDestroy(h);\n    return 0;\n}\n"


def main() -> int:
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "job": "p100_m3_base"}

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        report["error"] = "nvcc not found"
        print(json.dumps(report))
        (Path("/kaggle/working") / "p100_m3_base.json").write_text(json.dumps(report, indent=2))
        return 0
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    for line in ver.splitlines():
        if "release" in line:
            report["nvcc"] = line.strip()
    cu = WORK / "m3_base.cu"
    cu.write_text(CU)
    exe = WORK / "m3_base"
    tcc = time.time()
    r = subprocess.run([nvcc, "-O3", "-gencode", "arch=compute_60,code=sm_60",
                        "-lcublas", "-o", str(exe), str(cu)],
                       capture_output=True, text=True, timeout=1200)
    report["compile_s"] = round(time.time() - tcc, 1)
    if r.returncode != 0:
        report["error"] = "nvcc failed"
        report["nvcc_stderr_tail"] = r.stderr[-2000:]
        print(r.stderr[-2000:])
    else:
        p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=1800)
        print(p.stdout)
        if p.returncode != 0:
            report["error"] = f"exe rc={p.returncode}"
        report["exe_rc"] = p.returncode
        for line in p.stdout.splitlines():
            if line.startswith("JSON "):
                report["gemms"] = json.loads(line[5:])
            if line.startswith("device="):
                report["device"] = line
            if line.startswith("naive_vs_cublas"):
                report["naive_check"] = line
    report["wall_s"] = round(time.time() - t0, 1)
    out = Path("/kaggle/working") / "p100_m3_base.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[m3base] wall={report['wall_s']}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
