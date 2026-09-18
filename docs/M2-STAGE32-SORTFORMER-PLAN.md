# M2 Stage 3.2 — sortformer 组装 + 权重 arena 详细规划

状态：规划完成，**未写实现代码**（2026-09-18）。上游锚点：NeMo-Speech.cpp
a5b6953 浅克隆 `/tmp/nsc`。前置：3.1 已闭合（b905d07 GGUF/DFW1 reader）。
父计划：`docs/M2-STAGE3-ENDTOEND-PLAN.md` §3.2（本文是其展开）。

## 0. 范围

**目标**：本机纯 C++、零权重零 GPU，完成模型级组装——权重绑定（arena）+
per-chunk 前向 + host 状态机（FE→AOSC→BirthGate）首次连跑 + 中间抽头。
全部测试用 tiny 随机权重。

**非目标**（明确不做，防止范围蔓延）：
- 真权重对拍（3.3 K5）、生产 fixture 过门（3.4 K6）
- batch（B>1）——引擎永远 B=1，上游 attention/valid mask 仅服务于
  batch padding，scalar 路径无 mask（sortformer_model.cpp SortformerBatcher）
- `maybe_compact`/frozen segments（小时级长流的内存守卫；fixture ≤357s，
  probs_ ≤ 4467×4 floats，用不上）
- `speaker_for_frames`/word tagging（riva 面 API，parity 不需要）
- CUDA、性能优化、GGUF 写入器（生产侧资产，不动）
- f32bin_writer.py（DFW1 的 kernel 侧写入器，属 3.3 K5 prep）

## 1. 侦查结论（规划期从两侧代码钉死的事实，全部有出处）

### F1 组装序（上游 sortformer_model.cpp `SortformerGraph::build_graph` 逐行）

```
mel [n_mels, T_mel]
 → pre_encode stem (8x dw-striding, masked-tail 语义可选)  → chunk_embs [T3,512]
    ★ 无 xscale——AOSC 缓存契约：spkcache/fifo 存 raw pre-encode embeddings
 → concat over time [ spkcache | fifo | chunk_embs ]        → x [L,512], L=L1+L2+T3
 → xscale(sqrt(512)) 作用在整个 concat（含 state 前缀）
 → rel-pos 表 (2L-1 行, 公式现算) + 17 × conformer layer（full attention 无 mask）
 → encoder_proj Linear 512→192
 → 18 × transformer block（post-LN, 无 pos emb, ReLU FF, scale 1/sqrt(d_k)）
 → head: relu → Linear(192,192) → relu → Linear(192,4) → sigmoid
 → preds [L,4]（pre-gate）+ chunk_embs [T3,512]（第二个输出，喂 AOSC）
```

关键时序：**xscale 在 concat 之后**（所以 state 每进一个新 chunk 都被重新
scale——正确镜像 NeMo）；chunk_embs 输出必须在 xscale 前拷出。

### F2 尾块语义开关（Step 0 判决落进 API）

- 生产（C++ 上游）`run_chunk` 无 feat_len：stem 对整窗跑，尾行是"零垫
  mel 行上的真实 conv 输出"——**K6 必须复现的行为**。
- NeMo（PyTorch 真值）有 feat_len：MaskedConv 逐级掩码，尾行 = out.bias
  精确（v13 实锤）——**K5-A 必须复现的行为**。
- 两种模式下帧数几何相同（T3 = ceil(t_mel/8)，subsampled_len 整窗公式），
  只有尾行**值**不同 → 一个参数分流，host 状态机完全共用。
- 我们 `subsampling_forward` 已有 feat_len 参数（v13），`feat_len<=0` 即
  生产语义。引擎只需把它一路穿透。

### F3 host 驱动循环（上游 diar_pipeline.cpp `DiarStream`）

- chunk 调度（run_one_chunk）：`hop_mel = chunk_len×8`（streaming 160），
  `lc/rc = 0`（**两 preset 都是 0**，riva_streaming/riva_offline）；窗口
  = mel[stt-lc, end+rc) 边缘钳位；forced（非 final）块钳到 encoder 帧格，
  final_flush 允许任意尾长。
- 每步：`run_chunk(mel,t,spkcache,fifo)` → `aosc.update(chunk_embs,T3,
  preds,lc_enc,rc_enc)` → emitted → `birth_gate.append(emitted,probs_)` →
  缓冲裁剪（mel/audio 按下窗起点回退 n_fft/2）。
- lc_enc=lround、rc_enc=ceil（上游取整方向，逐字移植）。
- 流式 FE：`produce_new_mel_frames`（fe.cpp:714，~45 行调度器）——从
  audio_buf 增量产出完整 mel 帧；流首 reflect_left=true，此后
  reflect_left=false 从 `i_start*hop - n_fft/2` 起算。**我们 fe.cpp 没有
  这个函数**（只有全量 compute/compute_padded）→ 3.2c 前置小移植。
- `finish()`：补 n_fft/2 个几何衰减样本（`tail*=preemph` 循环），使
  post-preemphasis 恰为零，匹配 NeMo 尾帧 bit 级（a=0 退化全零）。
- offline 全窗入口（diarize_offline）：peak-normalize（max 只看正峰，
  上游 quirk 逐字保留）+ 一次 compute(reflect_left=true) + 单次
  run_chunk 空 state；t_enc>5000 loud 拒绝。

### F4 权重命名表（GGUF 名 = 运行时名 = NeMo state_dict 名经 remap）

conversion/diarization.py remap：`sortformer_modules.encoder_proj.*→
encoder_proj.*`、`sortformer_modules.*→head.*`、`transformer_encoder.*→
transformer.*`、`encoder.*` 原样。运行时模块 ctor 命名与之会合：

| 模块 | GGUF 张量名模式 | 数量 | dtype（3.1 已钉） |
|---|---|---|---|
| stem | `encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias}` + `encoder.pre_encode.out.{weight,bias}` | 12 | conv F16、out q8_0 |
| conformer×17 | `encoder.layers.{i}.norm_{feed_forward1,self_att,conv,feed_forward2,out}.{weight,bias}`、`.feed_forward{1,2}.linear{1,2}.{weight,bias}`、`.self_attn.{linear_q,linear_k,linear_v,linear_pos,linear_out}.{weight,bias}`（pos 无 bias）+ `.pos_bias_{u,v}`、`.conv.{pointwise_conv1,depthwise_conv,batch_norm,pointwise_conv2}.{weight,bias}` + `batch_norm.{running_mean,running_var}` | ~36×17 | linear q8_0、dw F16 |
| proj | `encoder_proj.{weight,bias}` | 2 | q8_0 |
| transformer×18 | `transformer.layers.{i}.first_sub_layer.{query_net,key_net,value_net,out_projection}.{weight,bias}`、`.layer_norm_{1,2}.{weight,bias}`、`.second_sub_layer.{dense_in,dense_out}.{weight,bias}` | 14×18 | q8_0 |
| head | `head.first_hidden_to_hidden.{weight,bias}`、`head.single_hidden_to_spks.{weight,bias}` | 4 | F32 |
| FE basis | `preprocessor.fb` | 1 | F32 |

stem 的 conv.{1,4} 是 ReLU（无张量），索引已从 K3 kernel 的 state_dict
加载代码实证（`conv.{0,2,3,5,6}`）。DFW1 用**同一套名**（写入器我们自己
写，3.3 落地），绑定层容器无关。

### F5 qkv 分离存储

GGUF 里 conformer self_attn 的 q/k/v/pos/out 是**分离张量**（上游 ggml
runtime 在 load 时才融合成 linear_qkv——fastconformer.cpp 注释明说
"stacked at load"）。我们的 `RelPosMhaWeights` 吃分离指针 → **直连，不做
融合**，少一个移植面。

### F6 PE 表运行时现算，不在权重文件里

NeMo RelPositionalEncoding 的 pe 不进 state_dict → GGUF 里也没有 →
上游 ggml 同样运行时算（get_pe_tensor）。引擎用 `relpos_table_forward`
公式现算（v12 已指纹证明与 NeMo 一致到 1.05e-6）。绑定层若发现任何疑似
PE 张量应 loud 报警（出现即规格变了，要人看）。

### F7 我们缺的件（3.2 的全部工作量所在）

| 件 | 现状 | 落点 |
|---|---|---|
| 权重绑定 arena | 无（reader 只有裸张量表） | 3.2a |
| 前向组装（F1 全链） | 无（层函数全齐） | 3.2b |
| 流式 FE 调度器 | 无（fe.cpp 只有全量） | 3.2c 前置 |
| engine 壳（F3 循环） | 无（AOSC/BirthGate/FE 全齐但从未连跑） | 3.2c |
| pre-gate timeline | 上游不保留（直接喂 BirthGate）；K5 需要 | 3.2c |

### F8 四 case → 引擎两个入口

v12-short-streaming / v13-mid-streaming / v13-mid-offline-preset →
流式入口（geometry streaming/offline_preset）；v12-mid-offline-full →
offline 全窗入口。offline_preset 仍是 AOSC 流式（geometry {312,100,100,100}）。

### F9 "arena"的定位修正

原计划口径"张量 arena"易读成 perf 优化。实际拆开：
- **权重 arena（进 3.2a 关键路径）**：reader 容器全量持有 fp32 数据，
  绑定层只做 名字→指针视图 + 形状断言 + 双向覆盖率 + 生命周期单一化
  （指针全部指进容器，禁拷贝）。正确性设施，不是优化。
- **scratch arena（不进 3.2）**：per-chunk 复用激活缓冲。现有层函数契约
  就是"自分配临时、参考实现速度无关"（layers.hpp 明文）。K5 CPU 粗算
  357s 音频 ~100-400 GFLOP，T4 4 核参考实现分钟级，可接受。若实测超预算
  再优化，不为它改层函数契约。

## 2. 三段提交（风险 #5 细化：神经前向先连，host 状态机后连）

### 3.2a 权重 arena + 绑定层

新增 `include/diar/sortformer.hpp` + `src/sortformer_weights.cpp`（或并入
sortformer.cpp，提交时定）：

```cpp
struct SortformerConfig {           // 上游 parse_config 的镜像，默认值即 v2 模型值
    int d_model = 512, encoder_layers = 17, n_heads = 8, d_ff = 2048,
        conv_kernel = 9, subsampling_factor = 8, conv_channels = 256,
        feat_in = 128, pos_emb_max_len = 5000;
    int transformer_layers = 18, hidden = 192, inner = 768, transformer_heads = 8;
    int num_speakers = 4;
    AoscScoringConfig scoring;      // sortformer.scoring.*（GGUF 可覆盖，缺省用默认）
};

class SortformerWeights {           // 权重 arena：拥有容器，发放指针视图
public:
    static SortformerWeights load(const std::string& path);   // magic 嗅探 GGUF/DFW1
    const SortformerConfig& config() const;
    const SubsamplingWeights& stem() const;
    const ConformerLayerWeights& conformer(int i) const;      // 17
    // encoder_proj / transformer(int i=18) / head 两组 Linear / mel_basis() ...
};
```

- 绑定规则：GGUF 路线 config 从 `sortformer.*` KV 读，**缺键取默认**
  （镜像上游 get_u32(key, default)，不是 fail——区别于 3.1 reader 的
  KV getter）；DFW1 无 KV → 用编译默认 + **形状驱动校验**（所有张量
  形状必须与默认 config 自洽，错即 loud，等价于把 config 错变成形状错）。
- 双向覆盖率：期望名集合（F4 表生成）vs 文件实际名集合，缺/多都 loud，
  报错带完整清单（沿用 3.1 fail-loud 风格）。
- 逐张量形状断言（[out,in] 等，按 hpp 契约）。
- 生命周期：`std::variant<GgufFile, F32WeightFile>` 按值持有，所有发放
  指针指进容器内 vector 的堆数据；禁二次拷贝（删除拷贝构造或文档钉死）。

测试 `tests/test_sortformer_weights.cpp`（~10 用例）：tiny GGUF + tiny
DFW1（复用 test_gguf.cpp 参考写入器模式，两容器同套名）→ 绑定全非空、
抽查指针数值（写入时埋可识别 pattern：q/k/v 指向不同数据，防 F5 类错）、
缺名/形状错/多余张量/坏 magic 各自 loud 且报对名字；GGUF KV 覆盖与缺省
两分支；DFW1 形状-默认 config 互证。

### 3.2b 前向组装 + 抽头

`src/sortformer.cpp`（组装主件）：

```cpp
struct SortformerChunkOutput {      // 镜像上游 ChunkOutput
    std::vector<float> preds;       // (L1+L2+T3)×n_spk，frame-major，pre-gate sigmoid
    std::vector<float> chunk_embs;  // T3×512，raw pre-encode（未 xscale，AOSC 契约）
    int total_frames = 0, chunk_frames = 0;
};

struct TapSink {                    // 3.3 分段定罪的抽头面
    virtual ~TapSink() = default;
    virtual void tap(const char* name, const float* data, int rows, int cols) = 0;
};  // 名单：stem.out / concat / xscaled / pos_emb / conformer.{i} / proj.out /
    // transformer.{i} / preds（逐层 conformer/transformer 可选开关，防 dump 爆）

SortformerChunkOutput sortformer_run_chunk(
    const float* mel, int t_mel, int feat_len,          // F2 开关：>0 NeMo / <=0 生产
    const float* spkcache, int spkcache_frames,
    const float* fifo, int fifo_frames,
    const SortformerWeights& w, TapSink* taps = nullptr);
```

实现 = F1 组装序逐行翻译成现有层函数调用（`subsampling_forward` →
手工 concat 缓冲 → `nn::xscale_forward` → `relpos_table_forward` →
`conformer_layer_forward`×17 → Linear 512→192 →
`transformer_block_forward`×18 → `diar_head_forward`）。每个边界一行
上游 pin 注释（文件+行为锚），沿用 Stage 1/2 风格。

测试 `tests/test_sortformer_forward.cpp`（~8 用例，tiny 随机权重 +
**非退化前置断言**——v13 教训成文）：
- 几何：t_mel=160→T3=20；86→11；带 state（L1=160,L2=80）→ L=260，
  preds 行数 260；pos 表 2L-1。
- 确定性双跑 bit 等；输入不被修改。
- chunk_embs 与 stem.out tap **bit 等**（未 xscale 契约的机器检查）；
  concat tap 已含 xscale（scale=sqrt(512) 抽查）。
- 空 state（首 chunk）与满 state 两形态。
- 双尾模式分叉：构造 86 帧窗、feat_len=80 → masked 尾行 == out.bias
  （bit 级，v13 语义）；同窗 feat_len=-1 → 尾行 ≠ out.bias 且有限。
- tap 名单覆盖测试（防改名漏抽）。

numpy 镜像抽取（"同源双实现"本地版）：把 K2/K3 kernel 里的
`pre_encode/relpos_table/relpos_mha/conformer_*/transformer_block/diar_head`
抽成 `reference/mirror_sortformer.py`（自包含、无 kaggle 依赖）+
`tests/test_stage3_mirror.py`：pytest 驱动一个 C++ dump 小工具
（`tools/`或 tests 内：tiny 权重+输入 → 跑 run_chunk → tap dump 落盘），
python 镜像同输入重算 → 逐 tap 比对，门 1e-5（与 K 门同级）。
镜像已在 K1/K2/K3 对 NeMo 证到 1e-5，此测试把证明传递到组装接线。

### 3.2c engine 壳：host 状态机首次连跑

前置小件：`produce_new_mel_frames` 移植进 fe.cpp（~45 行调度器 + 上游
语义注释），FE 差分 oracle 补一条流式用例（流首 reflect / 中段
reflect_left=false / 尾部保留不完整窗）。

新增 `include/diar/engine.hpp` + `src/engine.cpp`：

```cpp
struct EngineConfig {
    StreamGeometry geometry = StreamGeometry::streaming();
    bool nemo_tail_semantics = false;    // F2 路由：true=K5-A；false=K6/生产
    SegmentationConfig segmentation;
};

class DiarEngine {                        // 镜像 DiarStream（去掉 riva/compact 面）
public:
    DiarEngine(SortformerWeights weights, EngineConfig cfg);
    void feed_audio(const float* samples, std::size_t n);
    void finish();
    std::int64_t n_frames() const;
    const std::vector<float>& pre_gate_probs() const;   // AOSC emitted 直链（K5 面）
    const std::vector<float>& post_gate_probs() const;  // BirthGate timeline（K6 面）
    FrameProbabilities post_gate_frame_probabilities() const;
    std::vector<Segment> segments() const;              // upstream_segments_from_probs
    void set_tap_sink(TapSink*);                        // 转发给 run_chunk（K5 调试）
};
// 另：diarize_offline(...) 独立函数/静态入口（F3 offline 语义逐字：
// 正峰 normalize quirk + 单次全窗 run_chunk + t_enc>5000 loud）。
```

- run_one_chunk 调度逐字移植（hop/lc/rc/钳位/forced 帧格/final_flush/
  缓冲裁剪/finish 衰减尾/lround-ceil 取整）。
- pre-gate 链：上游把 emitted 直接喂 BirthGate 不保留；我们另存
  `raw_` 链（BirthGate.append 吃拷贝，已有契约）。
- `nemo_tail_semantics` 的 feat_len 规则：非 final chunk 全窗有效
  （mask no-op，传 t_mel 或 -1 等价）；final chunk 的 feat_len 钉死项
  见 T9。

测试 `tests/test_engine.cpp`（~10 用例）：
- **帧数账本守恒**（连跑核心不变式）：Σ emitted 帧数 == Σ(T3-lc-rc)；
  n_frames == 音频时长/0.08s 向上取整口径；spkcache/fifo 帧数演化曲线
  符合几何（fifo 涨到 80 后按 update_period=80 弹射进压缩、spkcache 封顶
  160）——M1 单步 oracle 之上加连跑账本。
- 短音频（不足一 hop）→ finish 后 final flush 出全部帧，无丢帧。
- finish 衰减尾：构造末样本非零音频，比对 last frame 与全量 offline
  计算的一致性口径（或断言 post-preemphasis 尾 pad 恰零——直接测
  produce_new_mel_frames 的 pad 语义）。
- 双跑确定性 bit 等；feed 分片任意切（1 样本粒度 vs 整块）结果相同
  （调度器状态机不依赖喂法——上游同性质，值得机器检查）。
- offline 入口：peak-normalize quirk 逐字（负峰-only 信号放大行为
  保留）+ 输出帧数 == (padded-nfft)/hop+1 口径。
- 双尾模式在 engine 层的分叉端到端可见（同音频两 config，final chunk
  帧 pre-gate 值不同）。

## 3. 尾块语义开关（K5/K6 分流）设计要点

- 单一参数 `feat_len`（run_chunk 层）+ 单一 bool（engine 层），语义注释
  引 Step 0 判决文档；**禁第三态**（"自动"之类）。
- 测试双向断言两模式确实分叉（防未来某次重构悄悄把 mask 删了而 K6 还绿）。
- T9（3.2c 钉死项）：K5-A final chunk 的 feat_len 精确值 = NeMo
  streaming_feat_loader 语义（m2-ref npz 有 per-chunk state_lens 佐证）。
  实现时从 npz 反推 + 写成测试常量，不拍脑袋。

## 4. 测试矩阵总表

| 文件 | 段 | 用例(估) | oracle 类型 |
|---|---|---|---|
| test_sortformer_weights.cpp | a | ~10 | tiny 双容器 + pattern 抽查 + loud 失败 |
| test_sortformer_forward.cpp | b | ~8 | 几何/确定性/契约断言/双尾分叉 |
| reference/mirror_sortformer.py + test_stage3_mirror.py | b | ~5 | numpy 镜像（K2/K3 血统）1e-5 |
| test_engine.cpp | c | ~10 | 账本守恒/调度不变式/双入口 |
| fe 流式 oracle 补充 | c | ~2 | 上游差分 |
| 合计 | | ~35 | pytest 预计 128→~150+ |

## 5. 风险表（3.2 专属；父计划风险表仍有效）

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 组装接线共因错误（C++ 与 numpy 镜像同作者，互绿≠对 NeMo 对） | 上游 build_graph 行号 pin 进每步注释；本地镜像对拍只声明"与已证镜像一致"；最终裁判=3.3 K5 chunk0 开环 1e-5 |
| R2 | 权重名/形状绑定错（F4 表手推） | 双向覆盖率 + 形状断言 + 名单测试钉死；stem 索引已从 K3 实证 |
| R3 | xscale 时机错（pre_encode 后提前 scale 会污染 AOSC 缓存） | 机器检查：chunk_embs tap bit-eq stem.out；concat tap 抽查 scale |
| R4 | qkv 融合/分离搞反（F5） | 绑定测试 pattern 抽查 q/k/v 指向不同数据 |
| R5 | 双尾模式被未来重构静默合并 | 分叉测试（两模式同输入必须不同 + masked 尾==out.bias bit 级） |
| R6 | 流式 FE 移植偏差（reflect/衰减尾） | produce_new_mel_frames 纳入 FE 差分 oracle；feed 分片不变性测试 |
| R7 | 账本 off-by-one（lc/rc 取整、final flush 几何） | 上游 lround/ceil 逐字；账本守恒测试 |
| R8 | 工程量超预算 | 三段各自全绿再下一步；镜像对拍可降级为首末层抽段 |
| R9 | K5 CPU 太慢 | 粗算分钟级可接受；scratch 优化后置不改层契约 |

## 6. 完成定义（DoD）

1. 3.2a/b/c 三个 commit，各自 ctest+pytest 全绿并 push，树干净。
2. tiny 权重端到端冒烟：流式 ≥4 chunk + finish 短尾 + offline 两入口，
   帧数账本闭合，双跑 bit 等，feed 分片不变。
3. 镜像对拍绿（tap dump vs reference/mirror_sortformer.py，≤1e-5）。
4. 双尾语义分叉测试绿。
5. 父计划 §3.2 挂本文链接；memory 关案。
6. 不含：真权重运行、K5/K6、CUDA、性能调优。

## 7. 工时口径

3.2a 半天（绑定+覆盖率是体力活，风险低）；3.2b 一天（组装 + 镜像抽取
+ 对拍占半天，是重头）；3.2c 半天到一天（调度移植 + 账本测试）。
合计 **2-3 天**——父计划"一到两天"偏乐观，按本文三段口径修正；若需压
回，R8 的降级开关是镜像对拍抽段化。
