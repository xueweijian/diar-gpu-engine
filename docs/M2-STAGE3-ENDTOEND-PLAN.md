# M2 Stage 3 — 端到端组装与 free-running 对拍规划

状态：Stage 0/1/2 已闭合（v13 三门全绿，2026-09-18，8ff41d0）。
本文是 Stage 3（M2 收尾）的开工规划。**只规划，不含实现。**

## 0. 决策记录（代替用户拍板）

1. **主线 = 端到端组装对拍，不是先修生产路径。** 理由：
   - ROADMAP M2 的出口本来就是"CPU/GPU 同 fixture 对比"，组装是引擎
     存在的意义；生产尾块补丁（选项 2）价值未证实，且长期被本引擎替代。
   - 掩码语义已封存在 `src/subsampling.cpp` + tests（可移植资产），
     不急于现在挪进 ggml 上游代码——跨代码库移植（ggml key-major
     布局）正是 v5-v13 里 bug 的温床。
2. **选项 2 降级为 Step 0 离线侦查**（零 GPU、零下载，数据全在本地）：
   先判定生产路径尾块到底有没有 bug，有则做一次最小独立补丁，无则归档关闭。
3. **命名对齐 ROADMAP**：日常口语的"M3 parity"在 ROADMAP 语境下是
   **M2 的收尾（Stage 3）**；ROADMAP M3 仍指 P100/CUDA 优化，本阶段
   完成后才解锁。
4. **权重路线 A（fp32 自有容器）先行，路线 B（q8 GGUF 直读）缓行**：
   A 语义最干净（与 NeMo 真值同精度），B 为 M3 前可选加固。

## 1. 现状资产盘点（Stage 3 的起点）

| 资产 | 位置 | 状态 |
|---|---|---|
| 手写算子核（LN/BN/conv/MHA/rel-shift/xscale…） | `src/nn.cpp` + 7 个层文件 | ✅ 双 oracle 测试 |
| 全链逐层 teacher-forced 对拍 | K1/K2/K3（v13 全绿 1e-5） | ✅ |
| host 侧 FE/AOSC/BirthGate/分段/hysteresis | `src/fe.cpp`/`aosc.cpp`/`birth_gate.cpp`/`diar.cpp` | ✅ 各自 oracle，**从未与神经核连跑** |
| NeMo fp32 参考（pre-gate total_preds 等） | `shared/diar-gpu-engine/m2-ref/*.npz` | ✅ short/mid |
| 生产 fixture（四 case，post-gate timeline） | `parity/fixtures/v12-*`, `v13-*` | ✅ loader 已验证 |
| **权重 loader / tensor arena** | 无 | ⬜ Stage 3.1 |
| **模型级组装（图 3.2）** | 无 | ⬜ Stage 3.2 |

## 2. Step 0 — 生产尾块侦查（本地离线，先行，~1-2h）

问题：ggml 生产二进制对 feat_len 非 32 倍数的尾块，是否也错了语义？

- **0a 钉 fixture 语义**：读 probdump patch 源码，确认 fixture 的
  `probs.f32` 是 pre-gate sigmoid 还是 post-gate（决定与 NeMo npz
  能否直接比；post-gate 则只做弱门比较或比 BirthGate 输入侧）。
- **0b 尾块对比**：m2-ref npz（pre-gate）vs 生产 probs，按 chunk 分层
  （首块/中块/尾块）统计逐帧差。T4/P100-era fixture 与 NeMo 参考的
  音频必须同源同几何（short/mid 与四 case 的对应关系先核 manifest）。
- **判定**：
  - 尾块差显著大于中块（>1e-2 量级）→ 生产确有尾 bug → 独立最小
    补丁（掩码语义移植进上游 subsample 路径，单独 commit，不与
    Stage 3 纠缠）。
  - 同分布 → 生产无辜 → 选项 2 归档关闭，理由写进本文件附录。
- 产出：`shared/diar-gpu-engine/m2-stage2/step0_production_tail_probe.md`。

> **Step 0 已完成（2026-09-18）**，判定：
> ①尾块 bug **静态成立**（run_chunk 无 feat_len 参数、subsampled_len 整窗
> ceil、stem 零掩码），当前 fixture 上**动力学潜伏**（两 case 尾块帧全静音，
> diff 0.0017-0.0003）；生产补丁降为独立小任务（先造"尾块带语音"fixture
> 再打补丁，规格见侦查报告附录），不在 Stage 3 关键路径。
> ②**K6 门必须走路线 B**：NeMo-fp32-vs-q8-生产在 spkcache 反馈环里随时长
> 复利到 0.6（mid gate-clean 区实测），fp32 引擎（路线 A）对拍 q8 fixture
> 的 max_abs≤0.05 门必然失败——与实现正确性无关。路线 A 降级为 K5 真值锚。

## 3. Stage 3 主线 — 五步

### 3.1 权重落地：q8_0 GGUF loader（主）+ fp32 容器（K5 锚）

- **主路线（K6 用）**：最小 q8_0 GGUF 读取器——读 header/张量表，
  q8_0 块（fp16 scale + 32×int8）反量化为 fp32 喂入现有算子（=
  M2 计划 §1 原意："CPU 参考核吃反量化后的权重做精确 FP32 数学"）。
  生产 GGUF（sha 0679cfeb…）只在 Kaggle 侧存在，不落本机。
- **锚路线（K5-A 用）**：kernel 内 .nemo → 自有 fp32 平铺 bin（layout
  与 include/diar/*.hpp 契约逐字一致），K5 运行时现场生成，不发布数据集。
- 本机测试：自写 tiny GGUF（q8_0 块写→读→反量化→逐字节比对）+
  fp32 bin round-trip + 头 schema 版本化（v1）+ 非退化前置断言
  （v13 教训：seed 777 类退化必须被测试前置抓住）。

### 3.2 张量 arena + 模型组装（本机纯 C++，无权重）

> **详细规划已展开**（2026-09-18）：`docs/M2-STAGE32-SORTFORMER-PLAN.md`
> ——侦查结论（组装序/尾块开关/host 循环/张量名表）、三段提交切分
> （3.2a 权重 arena → 3.2b 前向组装+抽头 → 3.2c engine 壳）、测试矩阵
> 与 DoD。工时口径修正为 2-3 天。本文节保留为概要。

- 新增 `src/sortformer.cpp` + `include/diar/sortformer.hpp`（暂名）：
  per-chunk 前向 mel → stem(masked, feat_len) → concat[spkcache|fifo|
  chunk_embs] → xscale+rel-pos → 17 conformer → encoder_proj →
  18 transformer → head → pre-gate preds (4,L)。
- **新增风险点 = host 状态机与神经核的第一次连跑**：AOSC/FIFO/
  spkcache 更新时序（M1 期各自 oracle 过，但组合语义未证）。
- 中间层抽头：每个边界留 optional dump 钩子（供 3.3 分段定罪）。
- 本机测试：tiny 权重端到端冒烟（形状/非退化/确定性双跑 bit 等）；
  分段 vs numpy 镜像（K2/K3 kernel 里的镜像代码抽成本地参考脚本，
  同源双实现哲学不变）。
- ggml 侧交叉验证仍按 M2 plan §0.2：NeMo 为真值源，ggml 二进制
  仅端到端双背书。

### 3.3 K5 kernel — 双路线对拍 NeMo（teacher→free 桥）

- 四 case 音频 → 本引擎（CPU fp32 编译）与 NeMo fp32 同机各跑一遍 →
  pre-gate frame_probs 逐帧比。**A/B 两跑**：
  - **K5-A（fp32 容器）= 真值锚**：干净比较。chunk0 开环应 1e-5 量级；
    chunkN 含 chunk_embs 反馈（闭环复利），门先测后钉分两档。
    若 chunk0 就崩 → 组装/接线 bug，用 3.2 的 dump 钩子分段定罪。
  - **K5-B（q8 GGUF）= 生产行为画像**：预期 ≈ Step 0 实测的生产-vs-NeMo
    包络（short ~0.1 / mid ~0.6，随时长复利）。门按 case 时长分档钉，
    用途是确认引擎复现生产行为（而非绝对真值）。
- Step 0 已证 fixture 与 NeMo 在 gate 折叠帧（未 establish 说话人清零）
  有范畴差——K5 比较一律用 pre-gate 双方，禁混 post-gate。
- 产出：verdict json 归档 + 阈值常量回填 tests。

### 3.4 K6 kernel — 过生产 fixture 双背书（M2 正式关门）

- 引擎输出 post-BirthGate timeline/probs vs `parity/fixtures/` 四 case。
- **必须跑路线 B（q8 GGUF）**：与 fixture 同权重才有紧门（Step 0 实测
  fp32-vs-q8 复利差 0.1-0.6，路线 A 在此门必死）。同权重同 host 语义下
  预期接近位级——先测后钉（预期 ≪0.05，若 >1e-2 即有实现分歧要查）。
- RTTM 段级对齐另记（边界 ±1 帧容忍策略，先测后钉）。
- 已知盲区：四 case 尾块帧全静音，尾块回归在此门不可见（Step 0 实测）；
  尾块覆盖由独立的生产补丁任务补（造语音尾 fixture）。
- 全绿 → **ROADMAP M2 exit**（FP32 参考后端完成），M3 CUDA sm_60
  解锁。

### 3.5 回填归档

- 阈值进 tests；verdict + stdout 归档 `shared/diar-gpu-engine/m2-stage3/`；
  ROADMAP/矩阵文档补节；memory 关案。

## 4. 风险表

| # | 风险 | 缓解 |
|---|---|---|
| 1 | spkcache 闭环误差复利超标 | 3.3 分 chunk 分层门；dump 钩子分段定罪 |
| 2 | 权重容器 layout 歧义 | 3.1 头 schema + round-trip 测试 + 逐张量断言 |
| 3 | fixture pre/post-gate 语义混比 | Step 0a 先钉死，禁直接比的铁律沿用 |
| 4 | host 状态机连跑组合 bug | 3.2 分段对拍；AOSC/BirthGate oracle 测试不删 |
| 5 | 工程量超预期（组装 > 预算） | 3.2 拆两段提交：神经前向先连，host 状态机后连 |
| 6 | CPU fp32 跑 357s 音频太慢 | T4 上 CPU 核够用；不行就分 chunk 进程并行，禁止为此提前进 CUDA |

## 5. 预算与铁律

- 本机：零权重、零大文件；代码 + tiny 随机权重测试。fp32 权重只存
  Kaggle 数据集。
- Kaggle：K5、K6 两个 kernel 轮次（+ 3.1 一个 dump 轮次），预算可控。
- 顺序：Step 0 → 3.1 → 3.2 → 3.3 → 3.4 → 3.5，单线推进，不并行开 CUDA。

## 6. 时间感（顺序工时，非日历承诺）

Step 0 半天内；3.1 半天；3.2 一到两天（最大块）；3.3/3.4 各一个
kernel 轮次（含可能的修一轮）；3.5 收尾半天。全绿后进度口径从 ~55%
改 ~75%（M0-M3 权重表）。
