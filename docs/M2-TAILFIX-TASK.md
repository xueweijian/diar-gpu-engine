# M2 独立任务：尾块语义修复（production + engine 对齐 NeMo）

来源：Step 0 侦查报告附录（`shared/diar-gpu-engine/m2-stage2/step0_production_tail_probe.md`）。
本文档是执行规格（2026-09-18 起草，补丁前置条件：tail fixture 落地）。

## 0. 任务是什么 / 不是什么

**是**：让尾 flush 块（feat_len 非 32 倍数的最后窗）的计算语义与 python NeMo
一致——NeMo 的 `MaskedConvSequential` 逐级乘性掩码 + 掩码行 fill。

**不是**：不是调参、不是放宽门；旧四 fixture 必须逐位不变（回归网，见 §4）。

## 1. 语义钉死（已有证据）

- NeMo 尾窗：loader 把窗口 **pad 到 32 的倍数**（如 86→96），`feat_length=86`；
  stem 三级卷积后按 `L1=ceil(86/2)=43, L2=22, L3=11` 逐级乘性掩码；
  掩码行输出 == `out.bias`（v13 K3 实证，本仓库 `src/subsampling.cpp` 的
  `feat_len>0` 分支即此语义）。
- production（a5b6953 ggml）：`run_chunk` 无 feat_len；窗口不做 padding；
  `subsampled_len` 对整窗做 ceil 链；stem 内零掩码。
- 两者在**尾部有效行**（末 1-2 行）分歧：K3 chunk035 row10 隐藏层差 34.8；
  现有四 fixture 因尾行静音而 probs 不可见（尾块 diff ≡0 / 0.0017）。

## 2. 关键约束（本任务新发现，起草时钉死）

1. **行数契约**：patch 后 production 的**发射行数不许变**（short 711 / mid
   4467 保持不变）。NeMo 的 npz 多 1-2 行是全零幻影行（T9 判决）；补丁
   应在 emit 层裁掉幻影行，只让**末 1-2 个有效行**的值从"错"变"对"。
   —— 否则四 fixture 的 wire 字节变化，回归网失效。
2. **窗口 padding 而非只传 feat_len**：引擎侧 `subsampling.cpp` 的
   `feat_len>0` 掩码分支已存在，但它作用在**未 padding 的短窗**上得到的是
   另一套几何。正确做法 = 先把尾窗 pad 到 32 倍数（对齐 NeMo loader），
   再走 `feat_len + 逐级掩码`，最后**裁掉幻影行**发射。
3. **两条路线开关已存在**：`sortformer_run_chunk(feat_len)`：
   `>0` = NeMo 掩码路线；`<=0` = production 全窗路线（K6 用）。补丁 = 
   让 stream 引擎在尾块走 `>0` 路线（engine.cpp 目前恒传 -1）。

## 3. 执行步骤（fixture 落地后）

1. **fixture 已就绪**：`kaggle/m2_tailfix`（本次新增，GPU P100 会话）产出
   `tail_slice.probs.f32`（production 前态）+ `tail_slice.nemo.npz`（NeMo 参考）
   + verdict（含首末行 diff 预览）。分析工具
   `scripts/analyze_tail_result.py`（尾 11 行单列）。
2. **引擎打补丁**（本地 + 单测，主要工作量）：
   - engine：尾块窗口 pad 到 32 倍数 + 传真实 feat_len + 发射裁行；
   - 单测：新增"尾块带语音"tiny 用例（合成），断言：补丁前 tail 行 ≠ NeMo 路线，
     补丁后 == NeMo 路线 && 行数不变；旧四 fixture 数值断言不变。
3. **Kaggle 对拍**：同一 tail 切片再跑一次 patched production（harness
   `apply_probdump_patch` 同款的 anchor patch，或直接在 engine 内验证）——
   `analyze_tail_result.py` 应显示 tail11 diff 从大值 → 与正文同量级。
4. **回归**：旧四 fixture 逐位不变（K6 门自动覆盖）；K5 重跑（尾行应更贴 NeMo）。
5. 单独 commit + 归档；不与 M2 Stage 3 混。

## 4. 验收标准

- 新 fixture：patch 后末 1-2 个有效行 vs NeMo 的 max_abs 落回正文水平
  （正文 = q8/状态复利量级，先测后钉；patch 前应显著大于它）。
- 旧四 fixture：probs.f32 逐字节不变（sha256 相同）。
- K5/K6 门全绿不回退。

## 5. 状态

- [x] Step 0 侦查（bug 静态成立 + 潜伏判定）
- [ ] tail fixture 产出（内核已推 RUNNING，见 `kaggle/m2_tailfix`）
- [ ] 引擎 patch（§3.2）
- [ ] Kaggle 前后对拍 + 回归
