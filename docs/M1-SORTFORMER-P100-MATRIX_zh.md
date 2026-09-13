# M1 — Sortformer v2 P100 实测矩阵（timing only，不含精度评分）

内核 `kaggle/sortformer_matrix` v4，状态 pass（2026-09-13，
04:23–05:09 UTC，约 46 分钟；其中构建 ~993s）。
v5（commit `5440375`）状态 pass（2026-09-13，05:55–06:42 UTC，
约 47 分钟；构建 ~976s；report schema v2）。
原始报告（Kaggle 产出物 `sortformer_matrix_report.json`，随内核输出存档，
按 `.gitignore` 约定不进 git，见 benchmarks/README 说明；
v5 报告另存 `/var/minis/shared/diar-gpu-engine/v5-sortformer_matrix_report.json`）；
12 条 schema 记录：`benchmarks/results/m1-sortformer-p100-matrix.jsonl`
（1–6 行 v4，7–12 行 v5；`commit` 字段 hardcode `d00a769` 为旧值，数字不受影响）。

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

## 5. 确定性：body 口径 verdict（v5 关闭 v4 confound）

v4 的 confound 已修：v5 用 segment body 比较
（`SPEAKER` 行的 start/duration/speaker，`%.3f` 格式化，
recording-id 字段剥离），whole-file sha 只作对照。结论：

| fixture | body 三跑 | 段数 | verdict |
|---|---|---|---|
| real_short（56.9s） | 1 个 hash | 9/9/9 | **确定**（whole-file 3 hash 纯属 recording-id 差异，反证修法有效） |
| real_mid（357s） | 3 个 hash | 40/40/38 | **非确定**（run2 少 2 段） |
| real_long（6432s） | 3 个 hash | 1415/1405/1391 | **非确定**（段数逐跑递减 10–14） |
| synth（300/900/1800s 空输出） | 平凡一致 | 0 | 非 informative，排除 |

关键细节：real_long 三跑段数 1415→1405→1391 逐跑递减、
mid run2（38 段）与 run0/1（40 段）分歧——v4 的"三跑段数一致
（9/38/1406）倾向主体确定"被 v5 推翻。注意 v5 real_long 段数
（~1400）与 v4（1406）接近但说话人数不同（v5 run0 报 1 speaker，
v4 报 2）——streaming 输出本身在跑间漂移，这是 parity 前提的硬伤。

**对 parity 工作的直接影响**：byte-identical 前提在短输入成立、
在中/长真实输入上不成立。M1 parity 门必须先回答"非确定的来源"
（候选：GPU 归约顺序 / AOSC 状态竞争 / chunk 边界条件），
否则任何 DER 对比都立不住——两次跑同一文件的差异可能大于
engine 改动的差异。

## 6. 并发：三变体 wall + chunk-vs-whole 拼接对照（v5 新增）

同 56.9s 音频切 4 块（每块 14.2s），三变体：

| 变体 | wall | RTF | 说明 |
|---|---|---|---|
| file 顺序基线（4 独立进程求和） | 3.10s | 0.0545 | 每次含完整进程启动+模型加载，最慢符合预期 |
| dir-c1（目录，concurrency 1） | 1.648s | 0.0290 | 同进程摊薄固定开销，已快 1.88x |
| dir-c4（目录，concurrency 4） | 1.325s | 0.0233 | 相对 c1 再快 **1.24x** |

两点结论：

1. **v4 的 1.27x 被复现（v5：1.24x）**——目录 batching 在单卡确有
   ~1.25x 收益，且 `bodies_match_file_baseline: true`（dir-c1/c4
   与 file 基线的逐文件 body 完全一致）：batching 不改变输出，
   可以放心用。
2. **chunk-vs-whole 不等价**：4 块偏移拼接（9 段）vs 整文件直跑
   （9 段），段数相同但 body sha 不同。切块改变 streaming 状态轨迹
   （边界效应），与"非确定"结论互相印证：streaming 输出对切分敏感。

给 engine 的设计含义：长音频切块+并发是速度解（c4 相对单进程基线
2.34x），但拼接 RTTM ≠ 整文件 RTTM——若走这条路，必须定义拼接
语义（边界重叠/投票）并进 parity 门单独记账，不能默认等价。

## 7. v5 附带观测

- offline 短输入 body 确定（两跑同 hash，9 段 4 spk）；
  offline 中输入两跑段数一致（34/34）但 body 不同——offline 也不完全确定。
- off/stream 分歧复现：short 4 spk vs 3 spk（与 v4 一致），
  mid 段数 34 vs 40（v4：34 vs 38——streaming 侧自身漂移，offline 侧稳定 34）。
- CPU 回退 short：wall 53.3s（RTF 0.938，约实时），GPU/CPU 比 27.4x；
  但 CPU 输出 body（9 段）与 GPU streaming body 不同——fallback 只保可用，不保一致。
- 长度拟合（v5 streaming 六点）：marginal RTF ≈ 0.050（~20x），
  截距 −11.05s（全量）/ −3.98s（仅真实）——负截距复现，
  超线性结论不变。

## 给 engine 设计的输入（M2 可用）

- P100 + Q8_0 + batch 1 streaming：短/中 ~30–34x 实时，长语音密集输入 ~21x。
  VRAM 常年 < 0.51 GiB——16GB 显存绰绰有余，batch/并发有空间。
- 长输入 GPU 等待（util 42%）+ 超线性成本 → 优先实验"切块+并发+拼接"，
  而不是先啃单文件流式优化。
- offline 快 1.5–1.8x 但改输出：速度分支与 parity 分支必须分开记账，
  不得以 RTTM 目测相似作为质量依据（沿用 M1 parity 验收门）。
- CPU fallback 约实时，可做无卡兜底。

## 下一步（按顺序）

1. ~~determinism 重测~~ ✓ v5 已做：短确定、中/长非确定（见 §5）。
   parity 门前置问题变为"非确定来源定位"（GPU 归约/AOSC 竞争/chunk 边界）。
2. ~~real_long 切块实验~~ ✓ v5 已做首版（见 §6）：切块+并发是速度解
   但拼接 ≠ 整文件，需定义拼接语义并单独过 parity。
   剩余归因：real_long 超线性中"长度 vs 内容"仍未完全分开
   （synth 无语音对照已有，缺"同长度不同语音密度"的中间点）。
3. offline vs streaming 的精度对照（DER/frame agreement），进 M1 parity 门——
   前提是先接受"同一几何内部亦有跑间漂移"（§5），精度门须用多次跑的分布而非单点。
4. 以上均为 timing/结构实验，不动 engine 实现。
