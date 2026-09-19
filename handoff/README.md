# M3 Step 6b 交接文档（手机 → 笔记本）

> 更新时间：2026-09-19 22:50（LCL）。Step 6 verdict 已收割（s6-red），Step 6b 代码已完成并待 CI/kernel 验证。
> **`git push` 不影响 Kaggle**；`kaggle kernels push` 会替换运行中会话——当前没有 RUNNING 的 kernel，推送窗口是打开的。

## Step 6 verdict 摘要（已归档 `shared/diar-gpu-engine/m3-step6/`）

- **s6-red，但数值面全绿**：G-B2 fp32 vs_cpu = 1.46e-06/4.77e-06/1.49e-06（门 0.05，~30000x 余量）→ route 数值透明；
  G-B fp32 vs fixture 与 CPU 逐位同数字（short 0.01088 / preset 0.1219 / stream 0.2906）；GPU det 全 True。
- **红在性能面**：G-D ms/chunk 292~1409 vs 门 25；mid-full cpu face 无指标（bench bug，已修）。
- **根因账本**（streaming fp32 315.1 ms/chunk）：**stem(CPU) = 236.8ms = 75%**；encoder_route 57.2（fp16 35.5）；head(CPU) 10.0；fe 摊 9.3。
- **重要修正**：route 57ms 不是 bug——引擎流式几何是 **L=296 全窗口**（spkcache160+fifo80+chunk20+context），Step4/5 kernel timeline 测的是 t=20/26，57ms 是 ~11x 行数的真实工作量。
- 计划层教训：stage1 的"stem 0.68%"是占比陷阱（40s CPU chunk 的占比）；绝对值 ~270ms 恒定，encoder 上 GPU 后变 75%。

## Step 6b 改动清单（本 commit，196 pytest + 17 ctest 全绿，CI 待验 nvcc）

1. **CudaStemRoute**（`include/diar/stem_cuda.hpp` + `src/stem_cuda.cpp` 新文件）：stem 全链上 GPU
   （im2col+GEMM、DW conv kernel、pointwise GEMM、flatten 转置、out linear），权重 device 常驻，
   v13 掩码语义逐位复刻（每 stage 激活后清行 + tail feat_len 输入掩码）。预计 ~1-2ms/chunk。
2. **head 进 route**：`CudaEncoderRoute` 构造加 `with_head`，`encoder_forward_headed` 在 device 上
   跑完 transformer 后直接 ReLU→hidden Linear→ReLU→spk Linear→sigmoid，**D2H 只拉 preds [L,S]**
   （省 10ms CPU head + 48x 回传带宽）。sortformer 侧 fallback 完整（拒载→CPU head）。
3. **bench full-offline 修复**：不再提前 return——honors --reps、算 determinism、算 vs_ref
   （mid-full cpu face 的 None/det-red 两个 reason 的根修）。
4. **PTX-JIT face（G-S6b-JIT）**：streaming 两 case 加 `fp32jit` route（`CUDA_FORCE_JIT=1`），
   compute_60 PTX-JIT 必须与 sm_75 SASS **bit-identical**——用户暂无 P100/V100，这是 Pascal 语义替身；
   G-E 跨卡互证改挂"用户硬件可用后补跑"（ge_t4_*.f32 已归档待对拍）。
5. **perf 对比表**：kernel 输出 engine ms/chunk vs 官方 45.5（archived m3-official-bench v4）每 route 一行 speedup。
6. 测试：`tests/test_stem_route.cpp`（delegating bit-eq / refusing fallback / constant geometry / taps-force-cpu /
   headed bit-eq / headed-refusal fallback），嵌入清单 38→40 blobs（+stem_cuda.hpp/cpp），k5/k6/m3 入口已重嵌。

## 下一步（拿到这台机器/笔记本后）

1. `git pull`，看 CI（nvcc 门首次编 stem_cuda.cpp——重点看它）
2. CI 绿 → `kaggle kernels push -p kaggle/m6_step6`（kernel v4 = step6b payload）
3. 预期：wall ~5h（CPU baseline 2 reps 仍是大头；JIT face +~10min）；等 COMPLETE → `python3 scripts/harvest_step6.py`
4. 判读重点：
   - G-D：stem GPU 后 streaming fp32 应 ~70ms（57 route + 9 fe + ~2 stem + ~1 head + host）；
     fp16 应 ~48ms。**仍超 25 门**——L=296 几何是硬工作量，Step 7 议题（fp16 attention/融合/CUDA Graph）。
   - G-B2 应仍 ~1e-6 级（stem/head GPU 化引入的 GEMM 重结合噪声会被 0.05 门吸收）
   - G-S6b-JIT：JIT vs SASS bit-identical（红=compute_60 PTX 语义回归，Step 3 后首检）
   - mid-full cpu face 应出指标（0.0x 级，与 K6 v5 v12-mid 0.0046 同族）
5. G-E：ge_t4_*.f32（3 个）在 `shared/diar-gpu-engine/m3-step6/`，等 P100/V100 可用后同 fixture 对拍。

## 关键文件

| 文件 | 内容 |
|---|---|
| `include/diar/stem_cuda.hpp` + `src/stem_cuda.cpp` | Step 6b stem GPU route（新） |
| `include/diar/encoder_cuda.hpp` + `src/encoder_cuda.cpp` | route + fused head |
| `include/diar/sortformer.hpp` + `src/sortformer.cpp` | StemRoute 接口 + 装配 |
| `kaggle/m6_step6/framework.py` | runner + judge()（JIT face + perf 表） |
| `tools/diar_bench_main.cpp` | bench（full-offline 修复 + stem/headed JSON 字段） |
| `scripts/harvest_step6.py` | 收割+归档（沿用） |
| `tools/build_step6_kernel.py` | kernel 嵌入清单（40 blobs） |
| `shared/diar-gpu-engine/m3-step6/` | Step 6 verdict + ge_t4_*.f32 归档 |

## 提醒

- fp16 门是 `√k` 随机游走（`1.5e-4·√k`），不是线性门。
- kernel 日志 `sh()` 是 capture_output——命令结束才 flush，中途只看启动行。
- 改任何被嵌入的源后：embed_stage3.py → build_k6_entry.py → build_m3_entry.py → build_step6_kernel.py 全部重跑（顺序！k6 读 k5 的产物）。
- 数值验证不在本机跑（无 CUDA）；本机只做 CPU 侧测试 + 语法。

## 机器可读摘要

- repo: `xueweijian/diar-gpu-engine` / branch: `main` / HEAD: 见 `git log -1`（Step 6b commit）
- kernel ref: `weijianxue/diar-m3-step6` / status: 无 RUNNING 会话（Step 6 已 COMPLETE 收割）
- next action: CI 绿 → `kaggle kernels push -p kaggle/m6_step6`（v4）→ ~5h → `python3 scripts/harvest_step6.py`
- step6 verdict: s6-red（数值全绿/perf 红）已归档；step6b 预期 G-D 仍红（~48-70ms vs 25）但 stem/head 归零是主验收点
