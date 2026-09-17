# M2 — 神经算子移植计划 (frame_probs 向上游对拍)

状态：M1 探针轨道已关案（v14，de092a8）。host 侧（FE/AOSC/BirthGate/
分段/hysteresis）全部 oracle 对拍完毕。`src/` 里神经核零行。
parity 真值：frame_probs 四 case 就绪，tolerance 门已钉
（frame_agreement≥0.999 / max_abs≤0.05 / mean_abs≤0.005）。

## 0. 已做的决策（代替用户拍板，不再讨论）

1. **手写 CPU 算子库，不 vendor ggml**。上游答案（NeMo fp32）只当
   teacher-forced 真值，不当运行时依赖。vendor ggml 等于把 M2 最该
   拥有的神经核外包，M3（CUDA sm_60）反正也要自有 kernel。
   自证风险不存在：门一律拿 NeMo 数字当 expected，不是拿自己当。
2. **NeMo = teacher-forced 真值源，ggml 二进制 = 端到端交叉验证**。
   每层吃 NeMo 参考输入、跟 NeMo 参考输出比；组装完成后 free-running
    frame_probs 再过一遍现有四 fixture（ggml 系输出），双背书。
3. **Stage 0 先 spike，NeMo 跑不起来就降格**：给 ggml
   `sortformer_model.cpp` 打第二枚 probdump 式 patch 吐中间层，
   对拍对象从 NeMo fp32 降格为 ggml q8，但 kernel 管线零新增依赖。
   spike 结论写进 §6 再开工 Stage 1。

## 1. 上游形状基线（a5b6953，GGUF sha `0679cfeb…da998a`）

per-chunk 图（`sortformer_model.h` 注释 + `build_graph`）：

```
mel (128, T_mel)
 -> NEST pre_encode（8x dw-striding conv stem）            -> (512, T3)，无 xscale（AOSC 存的就是 pre-scale 值）
 concat [ spkcache | fifo | chunk_embs ] (time 维)
 -> xscale (*sqrt(512)) + rel-pos + 17 conformer layers    (d_model 512, 8 头, d_ff 2048, conv k9, symmetric conv)
 -> encoder_proj 512->192
 -> 18 层 post-LN transformer                              (hidden 192, inner 768, 8 头, ReLU FF, 无 pos emb)
 -> head: relu -> Linear(192,192) -> relu -> Linear(192,4) -> sigmoid
 -> preds (4, L1+L2+T3) + chunk_embs (512, T3)
```

算子 pin（全部在上游源码逐字确认，不是假设）：

| 算子 | pin | 出处 |
|---|---|---|
| LayerNorm eps | `1e-5`（`ggml_norm(..., 1e-5)`） | `src/runtime/ggml/nn.cpp` LayerNorm::build_graph |
| BatchNorm1d(infer) eps | `1e-5`（host 上传常量） | 同文件 BatchNorm1d::set_data |
| attention scale | `1/sqrt(d_k)` | `rel_pos_attention.cpp:227`、`fastconformer.cpp:919` |
| xscale | `x * sqrt(d_model)`（NeMo PositionalEncoding.forward） | `fastconformer.cpp:1280` |
| conformer FF / conv 模块 | SiLU；conv 内 pointwise+GLU+depthwise+norm+pointwise | `fastconformer.cpp:236,374` + ConformerConv 注释 |
| transformer FF / head | ReLU；head relu→lin→relu→lin→sigmoid | `transformer_encoder.h` 注释 + `sortformer_model.cpp` build_graph §4 |
| rel-pos MHA | QKV 单 fused 投影 `[n_feat,3*n_feat]` + pos_bias_u/v；linear_pos 无 bias | `rel_pos_attention.h` 注释 |
| subsampling | 3 级 ceil-div /2（factor 8），对称 conv padding | `fastconformer.h` subsample_time_length |
| scoring 常量 | sil3 / thr 0.25 / boost 0.05 / sil_thr 0.2 / 0.75 / 1.5 / 0.5 | `sortformer_model.cpp` parse_config（GGUF `sortformer.scoring.*`，有默认值） |

注意：GGUF 是 **q8_0 量化权重**，NeMo 参考是 fp32。CPU 参考核吃反量化
后的权重做精确 FP32 数学，门阈值必须吸收量化差——阈值走"先测后钉"：
首次实测 spread 出来后 pin 死常量（同 FRAME_PROBS_TOLERANCE 哲学：
conservative 起步，随 M2 落地收紧，禁 env 调）。

## 2. Stage 0 — 参考捕获 spike（1 kernel，成败在此一举）

`kaggle/m2_refdump/` 新 kernel（与 matrix 不同的 slug，不混预算）。
输入：4 个 fixture 音频 + 同 matrix 几何（streaming preset）。
载体：vendor 上游 `dump_sortformer_reference.py` 的修改版
`dump_m2_reference.py`（放本仓，改动显式 diff），加两处：

1. **每 block 输出 hook**（deep chunk 才开）：17 conformer block +
   18 transformer block 的输出。量级：conformer 17×~300×512×4B≈10MB/
   chunk，transformer 18×~300×192×4B≈5MB/chunk，deep 取前 3 chunk +
   compression chunk，可接受。
2. 上游 dump 已有（直接复用）：`mel_window`、`pre_encode`、
   `fc_encoder`（=proj 后 transformer 输入）、`preds_full`、
   `chunk_preds`、state 快照、`total_preds`、geometry。

spike 必须回答（按顺序，任一 NO 即停，写 §6 结论）：

- S0-a：HF `nvidia/diar_streaming_sortformer_4spk-v2` 仓里有没有
  `.nemo` checkpoint 文件？（GGUF 同仓已证实可 curl，kernel 有 internet）
- S0-b：`pip install nemo_toolkit[asr]` 在 Kaggle 镜像里能否装完跑起来？
  （T4 sm_75，wheel 兼容性预期远好于 P100；体积 ~数 GB，注意 /tmp 配额）
- S0-c：hook 版 dump 脚本能否吐出全部中间层且 `total_preds` 与
  ggml 二进制同音频 preds 在 tolerance 内对上（跨实现 sanity）？

Fallback（S0 任一步失败）：ggml probdump 第二弹——patch
`sortformer_model.cpp` 逐 chunk 吐 pre_encode/fc/transformer/preds，
真值源降格为 ggml q8，Stage 2 门阈值相应收紧（同实现自比，spread 小）。

## 3. Stage 1 — CPU 张量核（本地+CI，零 GPU 预算）

`src/nn/`（新）：Linear、LayerNorm(eps 1e-5)、BatchNorm1d-infer、
softmax、SiLU、ReLU、sigmoid、Conv1d/depthwise-conv、GLU（channel
split）、rel-pos shift trick、sqrt-scale/xscale。全部 FP32，double
累加只用在单测 oracle 里（防自证：op 单测用 naive 独立实现或解析值，
如 softmax 归一性、LN 零均值单位方差、matmul 对三重循环）。

CI：合成小权重 op 级单测进 `ci.yml`（三 OS）。**真权重（147MB）永不
进 git**，真权重门一律 kernel 侧——既定纪律。

## 4. Stage 2 — 逐层 teacher-forced 对拍（M2 主体，3-4 kernel）

每层吃 `.npz` 参考输入、跟参考输出比，不依赖下层先做完。
按风险自下而上（.npz key 括号内）：

1. `head`（`fc_encoder`→trans-block 输出…若 hook 有 18 号 block 输出，
   head 输入即 transformer 输出；否则 head+transformer-stack 整体门，
   输入 `fc_encoder`、输出 `preds_full`）。trivial 先行，验证 harness。
2. `transformer block ×18`（hook 输出逐 block；post-LN 标准结构，
   无 pos emb）。单 block 门过 18 层全量。
3. `rel-pos MHA`（conformer/transformer 共用公式族，先读透
   `rel_pos_attention.cpp` shift trick；门藏在 block 门内，单独加
   synthetic 几何单测：mask 右对齐、padding -1e9）。
4. `ConformerConv`（pointwise+GLU+DW+BN/LN+pointwise+SiLU，最易错的是
   symmetric padding 与 cache 语义——离线 CacheMode::Disabled 只走
   symmetric 路，causal/cache-aware 是 M3 的事，**本阶段显式不做**）。
5. `ConformerFF`（macaron 半步，SiLU，trivial）。
6. `conformer layer ×17`（整块门，过 17 层全量）。
7. `pre_encode`（8x stem，输入 `mel_window` 输出 `pre_encode`）。

门阈值：首次实测后 pin 死（§1 量化差备注），收紧只能向前。

## 5. Stage 3 — 组装 + 端到端（1-2 kernel）

graph 组装（pre_encode→concat→17 conformer→proj→18
transformer→head）→ preds 门（teacher-forced state，输入全取 dump）；
然后 free-running：我们的 FE + 我们的 graph + 我们的 AOSC →
frame_probs，直接过现有 **4 fixture tolerance 门**。这一绿，
**M2 功能闭合**，manifest 升 schema v2（加中间层 tensor pin，
v1 fixture 保持有效，loader 向后兼容）。

## 6. Stage 4 — CUDA sm_60（即 M3 开头，不在本计划内展开）

P100 真机，FP32 先行；FP16-storage+FP32-accum 按 ADR-0002 跟上；
CUDA Graph 只在 streaming 几何稳定后（M3 §6）。

## 7. 预算与纪律

- kernel 预算：Stage 0（1）+ Stage 2（3-4）+ Stage 3（1-2）≈ 5-7 个，
  每个 ~50min；本地工作零 GPU 开销。
- 手机存储纪律：147MB GGUF 与 .nemo 永不进 PRoot，进 kernel input。
- 真值纪律：fixture 只能由 kernel 输出经 fill 脚本生成；gate 阈值
  pin 死常量；反自证（oracle 哲学）全程有效。
