# M3 前提变更：Kaggle P100 已退役 (2026-09-19 确认)

来源：Kaggle 官方公告
https://www.kaggle.com/product-announcements/735239
（LucyHe2, Kaggle Staff, 2026-08 发布）

- **P100 于 2026-09-15 从 Kaggle 退役**（Google Cloud 停供 P100，官方原文）。
- 替代 = **T4x2**：两块 T4，各 16GB；官方口径"T4 在多数负载上优于 P100"。
- 存量 P100 notebook 会被自动切到 T4x2。

## 对 M3 的影响（M3-P100-CUDA-PLAN 的前提修正）

1. **目标卡 P100 → T4（sm_75, Turing）**。CUDA 移植工作本身不变：
   算子集合、图、parity 门全部与卡无关；只有优化目标变。
2. **FP16-storage/FP32-compute 路线重新评估**：T4 有 tensor core
  （fp16 GEMM 实测 2-3x 于 fp32，见 p100_m3_base v3 数据
   ff_down 0.256→0.080 ms），但 fp16 输入量化会偏离 fp32 参考值；
   是否仍在 K6 的 0.05 max_abs 门内需实测定，初版先纯 fp32。
3. **双卡机会**（17 层 conformer 天然可切两卡）留到 Step 6，不进初期。
4. "machine_shape: NvidiaTeslaP100" 的历史之谜关闭：字段写法没错
   （CLI 2.2.4 内部同样发 machineShape=NvidiaTeslaP100），是无货被
   静默降级。tailfix kernel 的 device 实为 T4（m2-stage3/tailfx 的
   log 里 Tesla T4）——此前记忆中的"P100"是计划目标不是实跑硬件。

## 已到手的环境数据（T4, cc 7.5, 40 SM, CUDA 12.x + nvcc 就绪）

| GEMM | m×n×k | fp32 hot | fp16s hot |
|------|-------|----------|-----------|
| ff_up | 260×2048×512 | 0.254 ms | 0.132 ms |
| ff_down | 260×512×2048 | 0.256 ms | 0.080 ms |
| proj | 260×256×512 | 0.038 ms | 0.015 ms |

naive-vs-cuBLAS cross-check: max_abs = 0（harness 转置 bug 已修）。
CPU 引擎同形状 GEMM ~2.2 s/chunk（conformer 92.34% = 36.8 s/chunk），
T4 单卡 fp32 全 conformer ≈ 17 × (0.254+0.256+MHA+conv) ≈ 20-30 ms/chunk
量级 —— 三个数量级的余量，fp32 起步完全够。
