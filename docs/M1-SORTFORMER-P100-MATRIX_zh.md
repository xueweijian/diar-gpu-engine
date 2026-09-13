# M1 — Sortformer v2 P100 实测矩阵（timing only，不含精度评分）

内核 `kaggle/sortformer_matrix` v4，状态 pass（2026-09-13，
04:23–05:09 UTC，约 46 分钟；其中构建 ~993s）。
原始报告：`benchmarks/results/m1-sortformer-p100-matrix-report.json`；
6 条 schema 记录：`benchmarks/results/m1-sortformer-p100-matrix.jsonl`。

范围重申：纯说话人日志（no ASR/tokenizer/转写）。本轮只测时间，
DER/JER/frame agreement 一律未评分（`accuracy.status: not_scored`）。

## 版本 pin

- runtime：NeMo-Speech.cpp `a5b6953`，CUDA preset，`cuda_architectures=60`，
  CUDA toolkit 12.8，driver 580.159.04，`nemo-speech 0.1.0`
- 模型：`diar_streaming_sortformer_4spk-v2.q8_0.gguf`（147,075,776 B，
  sha256 `0679cfeb…66da99`），Q8_0
- 硬件：Tesla P100-PCIE-16GB，计算能力 6.0，SM 锁 1328 MHz
- harness commit：`d00a769`（`to_records` 所记 commit 字段）

## 1. GPU streaming 端到端（`--device cuda:0`，preset 无 = streaming，warmup 1 + 3 跑取中位）

| fixture | 音频 | wall 中位 | RTF | 实时倍率 | spread | SM util | VRAM 峰值 | segments/spk |
|---|---|---|---|---|---|---|---|---|
| real_short（4 人中文） | 56.9s | 1.97s | 0.0346 | 28.9x | 1.20% | 54.5% | 457 MiB | 9 / 3 |
| real_mid（视频音频） | 357.3s | 10.53s | 0.0295 | 33.9x | 0.74% | 91.0% | 457 MiB | 38 / 3 |
| real_long（mp3 转 16k 单声道） | 6432.4s（107min） | 308.03s | 0.0479 | 20.9x | 0.41% | 42.0% | 457 MiB | 1406 / 2 |
| synth_300s（无语音纯速度夹具） | 300s | 8.83s | 0.0294 | 34.0x | 0.08% | 91.5% | 457 MiB | 0 / 0 |
| synth_900s | 900s | 27.55s | 0.0306 | 32.7x | 0.55% | 82.0% | 457 MiB | 0 / 0 |
| synth_1800s | 1800s | 59.83s | 0.0332 | 30.1x | 0.87% | 72.0% | 457 MiB | 0 / 0 |

三点结论：

1. **三跑 spread 全 < 1.3%**——计时本身稳定，数字可复用为 engine 基线。
2. **长输入单位成本上升**：real_long 的 RTF（0.0479）比 mid（0.0295）高 ~60%。
   最小二乘 `wall = fixed + marginal*audio` 拟合给出**负截距**
   （全量 −10.49s / 仅真实 −3.71s，marginal RTF ≈ 0.0487/0.0484，
   即边际 ~20.5x）。截距为负在物理上不可能（固定开销不可能为负），
   这本身就是证据：**线性模型是错的，单位秒成本随输入变长而增长**
   （至少在语音密集的长输入上是超线性的）。
3. **利用率随之掉**：real_long 的 SM util 只有 42%，而 mid/synth_300s 是 91%。
   GPU 一半时间在等——瓶颈不在 GPU 算力，而在别处
   （候选：CPU 侧逐 chunk 后处理/AOSC 状态更新，或单文件流式状态增长）。
   同为长输入，synth_1800s（无语音）wall 59.8s vs real_long 308.0s：
   音频 3.6x，墙钟 5.2x——**有语音/有 segment 的内容显著更贵**，
   后处理随 segment 数（1406）增长是头号嫌疑。

## 2. offline 几何更便宜，但输出不一样（同一文件、同一卡）

| fixture | streaming wall/RTF | offline wall/RTF | offline util/VRAM | segments/spk（off vs stream） |
|---|---|---|---|---|
| real_short | 1.97s / 0.0346 | 1.26s / 0.0222（**1.56x 更快**） | 87% / 507 MiB | 9/4 vs 9/**3** |
| real_mid | 10.53s / 0.0295 | 5.97s / 0.0167（**1.76x 更快**） | 94% / 507 MiB | 34/3 vs 38/3 |

- offline 不仅更快，利用率也更高（87/94% vs 54/91%）——streaming 的
  逐 chunk 调度/状态管理有可观开销。
- **但两者 RTTM 不等价**：short 的说话人数 4 vs 3，mid 的段数 34 vs 38。
  几何形状改变聚类结果。engine 若用 offline 提速，必须按 M1 parity 计划
  重新过精度门（本轮未评分，此处只记录分歧存在）。

## 3. CPU 回退：同一二进制，约 26x 差距

short 在 `--device cpu` 下 wall 51.4s（RTF 0.904，约 1.1x 实时，
cpu/wall 比 3.93——多线程跑满）。GPU（1.97s）相对 CPU 加速 **26.1x**。
fallback 可用（不断言质量），应急可接受实时左右。

## 4. 目录并发：`-c 4` 相对 `-c 1` 加速 1.27x

同 56.9s 音频切 4 块：c1 wall 1.665s（RTF 0.0293）→ c4 wall 1.312s
（RTF 0.0231），4 文件全产出。单卡上目录级 batching 有收益但不大；
结合 §1 的 util 掉线，**长音频切块 + 并发**是下一个值得测的方向
（能同时验证"长度 vs 内容"的成本归因）。

## 5. 确定性：本轮方法学有 confound，结论待定

- 真实音频 3 fixture 各 3 跑 hash 全不同；合成（空输出）平凡一致。
- **但 harness 传了 `--recording-id <output.stem>`，而每跑输出路径不同
  （run0/run1/run2），RTTM 首字段 RECORDING-ID 必然不同 → sha 必然不同。**
  所以"3 unique hashes"不能解读为"非确定"。重测时必须去掉
  recording-id 差异后再比 segment 主体（或固定 recording-id）。
- 在 parity 工作开始前，这项必须重做——byte-identical 是 parity 前提。

## 给 engine 设计的输入（M2 可用）

- P100 + Q8_0 + batch 1 streaming：短/中 ~30–34x 实时，长语音密集输入 ~21x。
  VRAM 常年 < 0.51 GiB——16GB 显存绰绰有余，batch/并发有空间。
- 长输入 GPU 等待（util 42%）+ 超线性成本 → 优先实验"切块+并发+拼接"，
  而不是先啃单文件流式优化。
- offline 快 1.5–1.8x 但改输出：速度分支与 parity 分支必须分开记账，
  不得以 RTTM 目测相似作为质量依据（沿用 M1 parity 验收门）。
- CPU fallback 约实时，可做无卡兜底。

## 下一步（按顺序）

1. determinism 重测：固定 recording-id，比较 segment 主体字节一致。
2. real_long 切块实验：N 块之和 vs 整文件，对比 wall 与 util，
   把"长度"和"内容（含 1406 segments 后处理）"的成本分开。
3. offline vs streaming 的精度对照（DER/frame agreement），进 M1 parity 门。
4. 以上均为 timing/结构实验，不动 engine 实现。
