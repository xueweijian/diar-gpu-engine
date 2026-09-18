// M2 Stage 3.2a — SortformerWeights implementation (weight arena binding).
// See include/diar/sortformer.hpp for the full pin ledger.
#include "diar/sortformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace diar {
namespace {

[[noreturn]] void fail(const std::string& what) { throw WeightFileError(what); }

std::string shape_str(const std::vector<std::uint64_t>& s) {
    std::string r = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

// Canonical form for shape comparison: strip size-1 dims (see header — the
// converter squeezes some-but-not-all 1-dims and bytes are unaffected).
std::vector<std::uint64_t> canonical(std::vector<std::uint64_t> s) {
    std::vector<std::uint64_t> out;
    out.reserve(s.size());
    for (std::uint64_t d : s)
        if (d != 1) out.push_back(d);
    return out;
}

// name -> {data, canonical shape}, plus flag of which container it came from.
struct TensorIndex {
    std::map<std::string, std::pair<const float*, std::vector<std::uint64_t>>> map;
    bool from_gguf = false;

    static TensorIndex build(const std::variant<GgufFile, F32WeightFile>& f) {
        TensorIndex idx;
        if (const GgufFile* g = std::get_if<GgufFile>(&f)) {
            idx.from_gguf = true;
            for (const std::string& name : g->tensor_names()) {
                const GgufTensor& t = g->at(name);
                std::vector<std::uint64_t> row_major(t.ne.rbegin(), t.ne.rend());
                idx.map.emplace(name, std::make_pair(t.data.data(), canonical(std::move(row_major))));
            }
        } else {
            const F32WeightFile& w = std::get<F32WeightFile>(f);
            for (const std::string& name : w.tensor_names()) {
                const F32Tensor& t = w.at(name);
                idx.map.emplace(name,
                    std::make_pair(t.data.data(),
                        canonical(std::vector<std::uint64_t>(t.dims.begin(), t.dims.end()))));
            }
        }
        return idx;
    }
};

std::string read_magic(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) fail("sortformer weights: cannot open '" + path + "'");
    char magic[4] = {0, 0, 0, 0};
    const std::size_t got = std::fread(magic, 1, 4, fp);
    std::fclose(fp);
    if (got != 4) fail("sortformer weights: '" + path + "' shorter than a magic");
    return std::string(magic, 4);
}

// Both containers expose has_kv/kv_u32/kv_f32/kv_string with the same key
// namespace (GGUF sortformer.* KVs; DFW1 v2 config mirrors them).
template <class File>
SortformerConfig parse_config(const File& g) {
    SortformerConfig c;
    auto i32 = [&](const char* k, int d) {
        return g.has_kv(k) ? static_cast<int>(g.kv_u32(k)) : d;
    };
    auto boolean = [&](const char* k, bool d) {
        return g.has_kv(k) ? (g.kv_u32(k) != 0) : d;
    };
    auto f32 = [&](const char* k, float d) { return g.has_kv(k) ? g.kv_f32(k) : d; };

    c.d_model = i32("sortformer.encoder.d_model", 512);
    c.encoder_layers = i32("sortformer.encoder.n_layers", 17);
    c.encoder_heads = i32("sortformer.encoder.n_heads", 8);
    c.encoder_d_ff = i32("sortformer.encoder.d_ff", 2048);
    c.conv_kernel = i32("sortformer.encoder.conv_kernel_size", 9);
    c.subsampling_factor = i32("sortformer.encoder.subsampling_factor", 8);
    c.subsampling_conv_channels = i32("sortformer.encoder.subsampling_conv_channels", 256);
    c.feat_in = i32("sortformer.encoder.feat_in", 128);
    c.xscaling = boolean("sortformer.encoder.xscaling", true);
    c.pos_emb_max_len = i32("sortformer.encoder.pos_emb_max_len", 5000);
    c.transformer_layers = i32("sortformer.transformer.n_layers", 18);
    c.transformer_hidden = i32("sortformer.transformer.hidden_size", 192);
    c.transformer_inner = i32("sortformer.transformer.inner_size", 768);
    c.transformer_heads = i32("sortformer.transformer.n_heads", 8);
    c.num_speakers = i32("sortformer.num_speakers", 4);
    c.scoring.sil_frames_per_spk =
        i32("sortformer.scoring.spkcache_sil_frames_per_spk", 3);
    c.scoring.pred_score_threshold = f32("sortformer.scoring.pred_score_threshold", 0.25F);
    c.scoring.scores_boost_latest = f32("sortformer.scoring.scores_boost_latest", 0.05F);
    c.scoring.sil_threshold = f32("sortformer.scoring.sil_threshold", 0.2F);
    c.scoring.strong_boost_rate = f32("sortformer.scoring.strong_boost_rate", 0.75F);
    c.scoring.weak_boost_rate = f32("sortformer.scoring.weak_boost_rate", 1.5F);
    c.scoring.min_pos_scores_rate = f32("sortformer.scoring.min_pos_scores_rate", 0.5F);

    // ---- variant gates (the reference kernels implement exactly these) ----
    if (boolean("sortformer.encoder.use_bias", true) == false)
        fail("sortformer config: use_bias=false is not implemented (bias always on)");
    const std::string conv_norm =
        g.has_kv("sortformer.encoder.conv_norm") ? g.kv_string("sortformer.encoder.conv_norm")
                                                 : std::string("batch_norm");
    if (conv_norm != "batch_norm")
        fail("sortformer config: conv_norm='" + conv_norm + "' is not implemented (batch_norm only)");
    if (boolean("sortformer.transformer.pre_ln", false))
        fail("sortformer config: transformer pre_ln=true is not implemented (post-LN only)");

    // ---- preprocessor KVs must match the compiled-in FE pins ----
    const FrontendConfig fe{};
    const auto mismatch = [&](const char* what, const std::string& got,
                              const std::string& pin) {
        fail(std::string("sortformer config: preprocessor.") + what + "=" + got +
             " does not match the compiled-in frontend (" + pin +
             "); this engine's FE is not configurable");
    };
    const int sample_rate = i32("sortformer.preprocessor.sample_rate", fe.sample_rate);
    if (sample_rate != fe.sample_rate)
        mismatch("sample_rate", std::to_string(sample_rate), std::to_string(fe.sample_rate));
    const float window = f32("sortformer.preprocessor.window_size", fe.window_size_sec);
    if (window != fe.window_size_sec)
        mismatch("window_size", std::to_string(window), std::to_string(fe.window_size_sec));
    const float stride = f32("sortformer.preprocessor.window_stride", fe.window_stride_sec);
    if (stride != fe.window_stride_sec)
        mismatch("window_stride", std::to_string(stride), std::to_string(fe.window_stride_sec));
    const int n_fft = i32("sortformer.preprocessor.n_fft", fe.n_fft);
    if (n_fft != fe.n_fft)
        mismatch("n_fft", std::to_string(n_fft), std::to_string(fe.n_fft));
    const int n_mels = i32("sortformer.preprocessor.features", c.feat_in);
    if (n_mels != c.feat_in || n_mels != fe.n_mels)
        mismatch("features", std::to_string(n_mels), std::to_string(fe.n_mels) + " (== feat_in)");
    const float preemph = f32("sortformer.preprocessor.preemph", fe.preemph);
    if (preemph != fe.preemph)
        mismatch("preemph", std::to_string(preemph), std::to_string(fe.preemph));
    if (g.has_kv("sortformer.preprocessor.log_zero_guard") &&
        g.kv_f32("sortformer.preprocessor.log_zero_guard") != fe.log_zero_guard)
        mismatch("log_zero_guard", std::to_string(g.kv_f32("sortformer.preprocessor.log_zero_guard")),
            std::to_string(fe.log_zero_guard));
    // NOTE: sortformer.streaming.* geometry KVs are deliberately not parsed —
    // production presets override them unconditionally and the engine takes
    // explicit StreamGeometry (3.2c). sortformer_model.cpp:63 has the same note.

    // ---- structural sanity (shape formulas below rely on these) ----
    if (c.d_model <= 0 || c.d_model % c.encoder_heads != 0 || c.encoder_heads <= 0 ||
        c.transformer_hidden <= 0 || c.transformer_hidden % c.transformer_heads != 0 ||
        c.encoder_layers <= 0 || c.transformer_layers <= 0 || c.num_speakers <= 0 ||
        c.feat_in <= 0 || c.subsampling_conv_channels <= 0 || c.conv_kernel <= 0 ||
        c.encoder_d_ff <= 0 || c.transformer_inner <= 0 || c.pos_emb_max_len <= 0 ||
        c.subsampling_factor <= 0)
        fail("sortformer config: structurally invalid (divisibility/range)");
    return c;
}

}  // namespace

// Single source of truth for names+shapes+targets. w's per-layer vectors must
// already be sized. Member function: reads/writes the private view fields.
struct SortformerWeights::Slot {
    std::string name;
    std::vector<std::uint64_t> shape;  // canonical (size-1 dims stripped)
    const float** target;
};

std::vector<SortformerWeights::Slot> SortformerWeights::collect_slots(
    SortformerWeights* w, const SortformerConfig& c) {
    std::vector<Slot> s;
    auto add = [&](std::string name, std::vector<std::uint64_t> shape, const float** t) {
        s.push_back(Slot{std::move(name), std::move(shape), t});
    };
    const std::uint64_t D = c.d_model, C = c.subsampling_conv_channels,
                       F = c.feat_in, K = c.conv_kernel, H = c.encoder_heads,
                       DK = D / H, FF = c.encoder_d_ff, X = c.transformer_hidden,
                       I = c.transformer_inner, SPK = c.num_speakers;
    const std::uint64_t fbins = subsampling_freq_bins(c.feat_in, c.subsampling_factor);

    // ---- stem (encoder.pre_encode.*, indices conv.{0,2,3,5,6}) ----
    {
        SubsamplingWeights& t = w->stem_;
        add("encoder.pre_encode.conv.0.weight", {C, 3, 3}, &t.c0_w);
        add("encoder.pre_encode.conv.0.bias", {C}, &t.c0_b);
        add("encoder.pre_encode.conv.2.weight", {C, 3, 3}, &t.dw1_w);
        add("encoder.pre_encode.conv.2.bias", {C}, &t.dw1_b);
        add("encoder.pre_encode.conv.3.weight", {C, C}, &t.pw1_w);
        add("encoder.pre_encode.conv.3.bias", {C}, &t.pw1_b);
        add("encoder.pre_encode.conv.5.weight", {C, 3, 3}, &t.dw2_w);
        add("encoder.pre_encode.conv.5.bias", {C}, &t.dw2_b);
        add("encoder.pre_encode.conv.6.weight", {C, C}, &t.pw2_w);
        add("encoder.pre_encode.conv.6.bias", {C}, &t.pw2_b);
        add("encoder.pre_encode.out.weight", {D, C * fbins}, &t.out_w);
        add("encoder.pre_encode.out.bias", {D}, &t.out_b);
    }

    // ---- conformer layers (encoder.layers.{i}.*) ----
    for (int i = 0; i < c.encoder_layers; ++i) {
        ConformerLayerWeights& t = w->conf_[static_cast<std::size_t>(i)];
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        add(p + "norm_feed_forward1.weight", {D}, &t.n_ff1_g);
        add(p + "norm_feed_forward1.bias", {D}, &t.n_ff1_b);
        add(p + "feed_forward1.linear1.weight", {FF, D}, &t.ff1_w1);
        add(p + "feed_forward1.linear1.bias", {FF}, &t.ff1_b1);
        add(p + "feed_forward1.linear2.weight", {D, FF}, &t.ff1_w2);
        add(p + "feed_forward1.linear2.bias", {D}, &t.ff1_b2);
        add(p + "norm_self_att.weight", {D}, &t.n_sa_g);
        add(p + "norm_self_att.bias", {D}, &t.n_sa_b);
        add(p + "self_attn.linear_q.weight", {D, D}, &t.attn.q_w);
        add(p + "self_attn.linear_q.bias", {D}, &t.attn.q_b);
        add(p + "self_attn.linear_k.weight", {D, D}, &t.attn.k_w);
        add(p + "self_attn.linear_k.bias", {D}, &t.attn.k_b);
        add(p + "self_attn.linear_v.weight", {D, D}, &t.attn.v_w);
        add(p + "self_attn.linear_v.bias", {D}, &t.attn.v_b);
        add(p + "self_attn.linear_pos.weight", {D, D}, &t.attn.pos_w);
        add(p + "self_attn.pos_bias_u", {H, DK}, &t.attn.bu);
        add(p + "self_attn.pos_bias_v", {H, DK}, &t.attn.bv);
        add(p + "self_attn.linear_out.weight", {D, D}, &t.attn.out_w);
        add(p + "self_attn.linear_out.bias", {D}, &t.attn.out_b);
        add(p + "norm_conv.weight", {D}, &t.n_conv_g);
        add(p + "norm_conv.bias", {D}, &t.n_conv_b);
        add(p + "conv.pointwise_conv1.weight", {2 * D, D}, &t.conv.pw1_w);
        add(p + "conv.pointwise_conv1.bias", {2 * D}, &t.conv.pw1_b);
        add(p + "conv.depthwise_conv.weight", {D, K}, &t.conv.dw_w);
        add(p + "conv.depthwise_conv.bias", {D}, &t.conv.dw_b);
        add(p + "conv.batch_norm.weight", {D}, &t.conv.bn_w);
        add(p + "conv.batch_norm.bias", {D}, &t.conv.bn_b);
        add(p + "conv.batch_norm.running_mean", {D}, &t.conv.bn_mean);
        add(p + "conv.batch_norm.running_var", {D}, &t.conv.bn_var);
        add(p + "conv.pointwise_conv2.weight", {D, D}, &t.conv.pw2_w);
        add(p + "conv.pointwise_conv2.bias", {D}, &t.conv.pw2_b);
        add(p + "norm_feed_forward2.weight", {D}, &t.n_ff2_g);
        add(p + "norm_feed_forward2.bias", {D}, &t.n_ff2_b);
        add(p + "feed_forward2.linear1.weight", {FF, D}, &t.ff2_w1);
        add(p + "feed_forward2.linear1.bias", {FF}, &t.ff2_b1);
        add(p + "feed_forward2.linear2.weight", {D, FF}, &t.ff2_w2);
        add(p + "feed_forward2.linear2.bias", {D}, &t.ff2_b2);
        add(p + "norm_out.weight", {D}, &t.n_out_g);
        add(p + "norm_out.bias", {D}, &t.n_out_b);
    }

    // ---- encoder_proj + transformer (transformer.layers.{i}.*) ----
    add("encoder_proj.weight", {X, D}, &w->proj_w_);
    add("encoder_proj.bias", {X}, &w->proj_b_);
    for (int i = 0; i < c.transformer_layers; ++i) {
        TransformerBlockWeights& t = w->xf_[static_cast<std::size_t>(i)];
        const std::string p = "transformer.layers." + std::to_string(i) + ".";
        add(p + "first_sub_layer.query_net.weight", {X, X}, &t.q_w);
        add(p + "first_sub_layer.query_net.bias", {X}, &t.q_b);
        add(p + "first_sub_layer.key_net.weight", {X, X}, &t.k_w);
        add(p + "first_sub_layer.key_net.bias", {X}, &t.k_b);
        add(p + "first_sub_layer.value_net.weight", {X, X}, &t.v_w);
        add(p + "first_sub_layer.value_net.bias", {X}, &t.v_b);
        add(p + "first_sub_layer.out_projection.weight", {X, X}, &t.o_w);
        add(p + "first_sub_layer.out_projection.bias", {X}, &t.o_b);
        add(p + "layer_norm_1.weight", {X}, &t.ln1_g);
        add(p + "layer_norm_1.bias", {X}, &t.ln1_b);
        add(p + "second_sub_layer.dense_in.weight", {I, X}, &t.f1_w);
        add(p + "second_sub_layer.dense_in.bias", {I}, &t.f1_b);
        add(p + "second_sub_layer.dense_out.weight", {X, I}, &t.f2_w);
        add(p + "second_sub_layer.dense_out.bias", {X}, &t.f2_b);
        add(p + "layer_norm_2.weight", {X}, &t.ln2_g);
        add(p + "layer_norm_2.bias", {X}, &t.ln2_b);
    }

    // ---- head + mel basis ----
    add("head.first_hidden_to_hidden.weight", {X, X}, &w->head_w_);
    add("head.first_hidden_to_hidden.bias", {X}, &w->head_b_);
    add("head.single_hidden_to_spks.weight", {SPK, X}, &w->spks_w_);
    add("head.single_hidden_to_spks.bias", {SPK}, &w->spks_b_);
    add("preprocessor.fb", {F, FrontendConfig{}.n_fft / 2 + 1}, &w->fb_);
    return s;
}

int subsampling_freq_bins(int feat_in, int factor) {
    int stages = 0;
    for (int f = factor; f > 1; f >>= 1) {
        if ((f & 1) != 0) fail("subsampling factor " + std::to_string(factor) + " is not a power of two");
        ++stages;
    }
    int len = feat_in;
    for (int s = 0; s < stages; ++s) len = (len - 1) / 2 + 1;  // conv_out_len(k=3,s=2,p=1)
    return len;
}

const ConformerLayerWeights& SortformerWeights::conformer(int i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= conf_.size())
        throw std::out_of_range("conformer layer " + std::to_string(i));
    return conf_[static_cast<std::size_t>(i)];
}

const TransformerBlockWeights& SortformerWeights::transformer(int i) const {
    if (i < 0 || static_cast<std::size_t>(i) >= xf_.size())
        throw std::out_of_range("transformer layer " + std::to_string(i));
    return xf_[static_cast<std::size_t>(i)];
}

std::vector<std::string> SortformerWeights::expected_tensor_names(
    const SortformerConfig& cfg, bool require_pe) {
    SortformerWeights w;
    w.cfg_ = cfg;
    w.conf_.resize(static_cast<std::size_t>(cfg.encoder_layers));
    w.xf_.resize(static_cast<std::size_t>(cfg.transformer_layers));
    std::vector<std::string> names;
    for (const Slot& s : collect_slots(&w, cfg)) names.push_back(s.name);
    if (require_pe) names.push_back("encoder.pos_enc.pe");
    return names;
}

SortformerWeights SortformerWeights::load(const std::string& path) {
    const std::string magic = read_magic(path);
    SortformerWeights w;
    if (magic == "GGUF") {
        w.file_ = GgufFile::load(path);
        w.cfg_ = parse_config(std::get<GgufFile>(w.file_));
    } else if (magic == "DFW1") {
        w.file_ = F32WeightFile::load(path);
        w.cfg_ = parse_config(std::get<F32WeightFile>(w.file_));  // v2 config or defaults
    } else {
        std::string shown;
        for (char ch : magic) {
            static const char* hex = "0123456789abcdef";
            const unsigned char u = static_cast<unsigned char>(ch);
            shown += hex[u >> 4];
            shown += hex[u & 0xF];
        }
        fail("sortformer weights: unknown container magic 0x" + shown + " in '" + path +
             "' (expected 'GGUF' or 'DFW1')");
    }
    const bool gguf_route = std::holds_alternative<GgufFile>(w.file_);

    w.conf_.resize(static_cast<std::size_t>(w.cfg_.encoder_layers));
    w.xf_.resize(static_cast<std::size_t>(w.cfg_.transformer_layers));
    std::vector<Slot> slots = collect_slots(&w, w.cfg_);

    const TensorIndex idx = TensorIndex::build(w.file_);
    const std::uint64_t D = w.cfg_.d_model;

    // ---- bind + shape check (report ALL misses, not just the first) ----
    std::string missing;
    std::size_t bound = 0;
    for (const Slot& s : slots) {
        const auto it = idx.map.find(s.name);
        if (it == idx.map.end()) {
            if (!missing.empty()) missing += ", ";
            missing += s.name;
            continue;
        }
        if (it->second.second != s.shape)
            fail("sortformer weights: tensor '" + s.name + "' has shape " +
                 shape_str(it->second.second) + ", expected " + shape_str(s.shape));
        *s.target = it->second.first;
        ++bound;
    }

    // All expected tensors present? (full list, not first-miss)
    if (!missing.empty())
        fail("sortformer weights: missing tensors: " + missing);

    // ---- PE route rule ----
    const std::vector<std::uint64_t> pe_shape = {2u * w.cfg_.pos_emb_max_len - 1u, D};
    const auto pe_it = idx.map.find("encoder.pos_enc.pe");
    if (pe_it != idx.map.end()) {
        if (pe_it->second.second != pe_shape)
            fail("sortformer weights: encoder.pos_enc.pe has shape " +
                 shape_str(pe_it->second.second) + ", expected " + shape_str(pe_shape));
        w.pe_ = pe_it->second.first;
        w.pe_rows_ = static_cast<int>(pe_shape[0]);
        ++bound;
    } else if (gguf_route) {
        fail("sortformer weights: GGUF route requires encoder.pos_enc.pe "
             "(production loads the stored table; rel_pos_attention.cpp:119)");
    }

    // ---- reverse coverage: everything in the file must be expected ----
    std::set<std::string> expected;
    for (const Slot& s : slots) expected.insert(s.name);
    expected.insert("encoder.pos_enc.pe");  // bound-if-present above
    std::string extras;
    for (const auto& entry : idx.map) {
        if (expected.count(entry.first)) continue;
        if (!extras.empty()) extras += ", ";
        extras += entry.first;
    }
    if (!extras.empty())
        fail("sortformer weights: unexpected tensors (name typo or foreign model?): " + extras);

    w.bound_ = bound;
    return w;
}



// ---------------------------------------------------------------------------
// 3.2b forward assembly — order pinned in sortformer.hpp.
int sortformer_subsampled_len(int t_mel, int subsampling_factor) {
    if (t_mel <= 0) return 0;  // C++ trunc-div would say 1 here; python floor says 0
    int stages = 0;
    for (int f = subsampling_factor; f > 1; f >>= 1) ++stages;
    int len = t_mel;
    for (int s = 0; s < stages; ++s) len = (len + 2 * 1 - 3) / 2 + 1;  // k3 s2 p1
    return len;
}

SortformerChunkOutput sortformer_run_chunk(const float* mel, int t_mel, int feat_len,
    const float* spkcache, int spkcache_frames, const float* fifo, int fifo_frames,
    const SortformerWeights& w, TapSink* taps) {
    const SortformerConfig& c = w.config();
    const int D = c.d_model, X = c.transformer_hidden, SPK = c.num_speakers;
    const int T3 = sortformer_subsampled_len(t_mel, c.subsampling_factor);
    const int L = spkcache_frames + fifo_frames + T3;

    SortformerChunkOutput out;
    out.total_frames = L;
    out.chunk_frames = T3;
    if (L <= 0) return out;

    // 1) stem — raw pre-encode embeddings (chunk_embs BEFORE any scaling)
    out.chunk_embs.resize(static_cast<std::size_t>(T3) * D);
    if (T3 > 0)
        subsampling_forward(mel, w.stem(), out.chunk_embs.data(), t_mel, c.feat_in,
            c.subsampling_conv_channels, D, feat_len);
    if (taps && T3 > 0) taps->tap("stem.out", out.chunk_embs.data(), T3, D);

    // 2) concat [spkcache | fifo | chunk]
    std::vector<float> x(static_cast<std::size_t>(L) * D);
    {
        float* dst = x.data();
        if (spkcache_frames > 0) {
            std::memcpy(dst, spkcache, sizeof(float) * spkcache_frames * D);
            dst += static_cast<std::size_t>(spkcache_frames) * D;
        }
        if (fifo_frames > 0) {
            std::memcpy(dst, fifo, sizeof(float) * fifo_frames * D);
            dst += static_cast<std::size_t>(fifo_frames) * D;
        }
        if (T3 > 0)
            std::memcpy(dst, out.chunk_embs.data(), sizeof(float) * T3 * D);
    }
    if (taps) taps->tap("concat.raw", x.data(), L, D);

    // 3) xscale on the whole concat (state prefix included)
    if (c.xscaling) {
        const float scale = std::sqrt(static_cast<float>(D));
        for (std::size_t i = 0; i < x.size(); ++i) x[i] *= scale;  // element-wise
        if (taps) taps->tap("xscaled", x.data(), L, D);
    }

    // 4) rel-pos table: slice the stored GGUF table exactly like production,
    //    or rebuild by the pinned formula (DFW1 route).
    if (L > c.pos_emb_max_len)
        throw std::invalid_argument(
            "sortformer_run_chunk: L=" + std::to_string(L) + " exceeds pos_emb_max_len=" +
            std::to_string(c.pos_emb_max_len) + " (upstream validate_stream_geometry guard)");
    std::vector<float> pe(static_cast<std::size_t>(2 * L - 1) * D);
    if (w.pe_table() != nullptr) {
        const int center = w.pe_rows() / 2 + 1;  // v12 live probe pin
        std::memcpy(pe.data(), w.pe_table() + static_cast<std::size_t>(center - L) * D,
            sizeof(float) * (2 * L - 1) * D);
    } else {
        relpos_table_forward(pe.data(), L, D);
    }
    if (taps) taps->tap("pos_emb", pe.data(), 2 * L - 1, D);

    // 5) conformer chain (ping-pong)
    std::vector<float> y(static_cast<std::size_t>(L) * D);
    float* cur = x.data();
    float* nxt = y.data();
    for (int i = 0; i < c.encoder_layers; ++i) {
        conformer_layer_forward(cur, pe.data(), w.conformer(i), nxt, L, D, c.encoder_d_ff,
            c.encoder_heads, c.conv_kernel);
        std::swap(cur, nxt);
        if (taps) {
            char name[32];
            std::snprintf(name, sizeof(name), "conformer.%d", i);
            taps->tap(name, cur, L, D);
        }
    }

    // 6) encoder_proj
    std::vector<float> px(static_cast<std::size_t>(L) * X);
    nn::linear_forward(cur, w.proj_w(), w.proj_b(), px.data(), static_cast<std::size_t>(L),
        static_cast<std::size_t>(D), static_cast<std::size_t>(X));
    if (taps) taps->tap("proj.out", px.data(), L, X);

    // 7) transformer chain (post-LN)
    std::vector<float> py(static_cast<std::size_t>(L) * X);
    cur = px.data();
    nxt = py.data();
    for (int i = 0; i < c.transformer_layers; ++i) {
        transformer_block_forward(cur, w.transformer(i), nxt, L, X, c.transformer_inner,
            c.transformer_heads);
        std::swap(cur, nxt);
        if (taps) {
            char name[32];
            std::snprintf(name, sizeof(name), "transformer.%d", i);
            taps->tap(name, cur, L, X);
        }
    }

    // 8) head -> pre-gate preds
    out.preds.resize(static_cast<std::size_t>(L) * SPK);
    diar_head_forward(cur, w.head_hidden_w(), w.head_hidden_b(), w.head_spks_w(),
        w.head_spks_b(), out.preds.data(), L, X, SPK);
    if (taps) taps->tap("preds", out.preds.data(), L, SPK);
    return out;
}

}  // namespace diar
