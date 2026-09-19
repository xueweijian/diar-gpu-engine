# M3 Step 6 交接文档（手机 → 笔记本）

> 更新时间：2026-09-19 17:23（LCL）。手机没电关机，笔记本 `git pull` 后继续。
> **先说最重要的：`git push` 不会影响 Kaggle。** Kaggle 会话只会被 `kaggle kernels push` 替换，
> `git push` 只触发 CI 编译/测试（`ci.yml`），`kaggle-p100.yml` 是纯手动 `workflow_dispatch`。
> 所以这个文档现在就可以 push。

## 最少背景

- 仓库 `xueweijian/diar-gpu-engine`，M3 Step 6：Stage 1-5 的 CPU/CUDA 组件做引擎级端到端集成，
  Kaggle T4 上 4 fixture 验收（单 fatbin + diar-bench + 跨卡互证形态）。
- 手机（PRoot）只做源码/小验证；重活（`.nemo` 权重、4 fixture bench）全在 Kaggle kernel。
- Step 5 已验证 fp16 权重存储链；Step 6 目标是完整交付形态。

## 当前精确状态（17:23 LCL）

### git / 本地
- `main`，HEAD `3a96990`（`scripts/harvest_step6.py` 收割脚本；运行内容等价 kernel 嵌入源）。
- kernel v3 嵌入源码 = `15f3a8e`，与 HEAD 运行等价。
- 本次更新：`handoff/README.md`（本文件），commit 后 push，`origin/main` 应对齐。
- `git status` 之前只有 `?? handoff/`，其余干净。

### Kaggle kernel
- ref: `weijianxue/diar-m3-step6`，状态 **RUNNING**（17:23 live 日志确认）。
- 编译：`nvcc` 过，fatbin `sm_60/sm_70/sm_75 + ptx60`，指纹 `T4 / cc7.5 / driver13000 / cublas120804`。
- 错误关键词（Traceback/Error/CUDA error/OOM/Killed）0 命中，到目前所有 bench `rc=0`。
- 进度（wall 自 kernel 启动）：
  - `2.0s` sources 38 物化；`16.8s` CPU selftest + diar-bench selftest OK
  - `105.5s` nvcc+fatbin；`108s` 起 4 fixture 并行：
    `v12-short-streaming-r0` / `v12-mid-offline-full-r0`（4467帧）/
    `v13-mid-offline-preset-r0` / `v13-mid-streaming-r0`
  - `2665s` short cpu 完：2 reps ~35.3-35.5s/chunk，
    `vs_ref max 0.0108817 / mean 1.5e-4 / agree 0.999648 / rows_delta 0`
  - `2713s` short fp32 完：3 reps best **383.2ms/chunk**，
    `determinism true`，`vs_ref max 0.0108831`（与 cpu 只差 1.4e-06，G-B2 早期信号绿）
  - `2757s` short fp16 完：3 reps best **328.7ms/chunk**，
    `determinism true`，`vs_ref max 0.0118192`
  - `3021s` mid-full cpu 完：wall 2909s（~48min）
  - 剩余：两个 mid 的 cpu baseline（最慢大头）+ 各自 gpu reps（分钟级）。
- ETA：总墙钟 2-3h，**18:00-18:30 左右出 verdict**；20:30 有 follow-up 收割任务（session `bd30e7f3`）自动拉。

### CI
- `15f3a8e` 全绿 6/6（asan / unit×3 / hygiene / nvcc-compile-gate）。
- `3a96990` 只加收割脚本；之前一次查 CI 被沙箱 DNS 抖动打断（`api.github.com` 解析失败），非代码问题。
  笔记本上可重查：`gh api repos/xueweijian/diar-gpu-engine/actions/runs?head_sha=$(git rev-parse HEAD)`。

### 门早期信号（verdict 出来前只看趋势，不下结论）
- G-B（route vs fixture，HARD max<=0.05）：short 三路线 0.0108-0.0118，绿区。
- G-B2（fp32 vs_cpu<=0.05，**门控项**）：short fp32 vs cpu 差 1.4e-06，绿信号；等 `judge()` 的 `vs_cpu_max_abs` 落 verdict。
- G-C（determinism）：short 3/3 `bit-identical true`。
- G-D（routed ms/chunk<=25，官方 45.5）：short fp32 383 / fp16 328，**按当前数是 red**；
  瓶颈看 profile（此前信号 stem 占比大；在笔记本上看 `route_effective` / `route_refused` 再定是路由还是算子问题）。
- fp16：advisory，不 hard fail。

## 今天修过的三个坑（已进树，不用重修）

1. `ci.yml` heredoc 吞 `nvcc` 行前缀 → CI 非法命令（`fb209d7` 修）。
2. kernel 嵌入清单漏 `diar/profile.hpp` → g++ 死（38 blobs，builder+framework+测试计数同步，`fb209d7` 修）。
3. `dev_linear` 链接错：hpp 两个 NON-template 重载声明 vs cpp `template<WT>`+显式实例化 →
   CI nvcc 门 + Kaggle v2 同报 `undefined reference`（`15f3a8e` 修：hpp 改 template+extern template 对）。

## 笔记本接手步骤

1. `git pull`，确认 HEAD（含本文件）。
2. 查 kernel：`python3 scripts/kaggle_via_ip.py kernels-status --ref weijianxue/diar-m3-step6`
3. COMPLETE 后：`python3 scripts/harvest_step6.py`
   产物进 `shared/diar-gpu-engine/m3-step6/`（verdict 表 + stdout 归档）。
4. 判读门（实现 `kaggle/m6_step6/framework.py::judge`）：
   HARD `max 0.05 / mean 0.005 / agree 0.999`；G-D `<=25ms`（full-offline 跳过）；
   `route_refused!=0` 直接 red；`vs_cpu_max_abs>0.05` red（G-B2 门控）。
5. 若 G-D red：先看 `route_refused`+`route_effective`，再看 profile 里 stem/fe 是否走 cpu。
6. G-E 跨卡互证：verdict 后按 P100/V100 实际硬件跑 probs max_abs 互证（`<=1e-6`）。
7. 需要的凭证：`GH_TOKEN`（GitHub API）、`KAGGLE_API_TOKEN`（kaggle_via_ip.py / Kaggle CLI）。

## 先读这几个文件

| 文件 | 为什么 |
|---|---|
| `docs/M3-CUDA-PLAN.md` | Step 6 验收标准 |
| `kaggle/m6_step6/framework.py` | runner + `judge()` 门定义 |
| `tools/diar_bench_main.cpp` | diar-bench CLI 与输出格式 |
| `include/diar/backend.hpp` | Step 6 抽象接口 |
| `src/backend_cuda.cpp` | CUDA 实现 |
| `scripts/harvest_step6.py` | 收割+归档脚本 |
| `tools/build_step6_kernel.py` | kernel 嵌入清单（38 blobs，单事实源） |

## 提醒

- `git push` 安全；危险的是 `kaggle kernels push -p kaggle/m6_step6`（会替换 RUNNING 会话）——verdict 出来前别碰。
- fp16 门是 `√k` 随机游走（`1.5e-4·√k`），不是线性门。
- kernel 日志 `sh()` 是 capture_output——命令结束才 flush，中途只看启动行。

## 机器可读摘要

- repo: `xueweijian/diar-gpu-engine` / branch: `main` / HEAD: `3a96990` + handoff update（push 后笔记本 `git pull` 核对）
- kernel ref: `weijianxue/diar-m3-step6` / status: `RUNNING` (17:23 LCL)
- next action: COMPLETE → `python3 scripts/harvest_step6.py`
- early signals: G-B green / G-B2 likely-green / G-C 3-3 true / G-D red-trend (383/328ms vs 25ms)
