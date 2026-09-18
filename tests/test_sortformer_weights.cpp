// M2 Stage 3.2a — SortformerWeights (weight arena) tests.
//
// Fixtures are written by independent byte-level writers in this file (same
// discipline as test_gguf.cpp): the tensor table below is hand-derived from
// the documented GGUF schema (conversion/diarization.py remap + module ctor
// names), NEVER from SortformerWeights' own expected-name table — the test
// must be able to catch name/shape typos on either side.
//
// Tiny config keeps every tensor small; feat_in stays at the frontend pin
// 128 because the binding enforces the FE pin set (only preprocessor.fb
// [128,257] and out_w's second dim grow with it — both trivial).
// Expected counts for the tiny config: stem 12 + conformer 2x39 + proj 2 +
// transformer 2x16 + head 4 + fb 1 = 129 (+1 pe on the GGUF route).
//
// Value oracle: tensor #ti fills element j with 0.5*ti + 0.001*j, so a bound
// pointer is verified by recomputing (ti, j) from the fixture table.

#include "diar/sortformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    expect(std::fabs(actual - expected) <= tolerance,
        message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

void expect_throws_substr(const char* needle,
    const std::function<void()>& fn, const std::string& message) {
    try {
        fn();
    } catch (const std::exception& e) {
        expect(std::string(e.what()).find(needle) != std::string::npos,
            message + " (message was: " + e.what() + ")");
        return;
    }
    expect(false, message + " (no throw)");
}

// ---------------------------------------------------------------- writers
void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; i++) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_str(std::vector<std::uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void put_f32(std::vector<std::uint8_t>& b, float f) {
    std::uint32_t v;
    std::memcpy(&v, &f, 4);
    put_u32(b, v);
}

struct TSpec {
    std::string name;
    std::vector<std::uint32_t> dims;  // row-major, outermost first
};

std::size_t numel(const TSpec& t) {
    std::size_t n = 1;
    for (std::uint32_t d : t.dims) n *= d;
    return n;
}

float pattern(std::size_t tensor_index, std::size_t element) {
    return 0.5F * static_cast<float>(tensor_index) + 0.001F * static_cast<float>(element);
}

std::string build_gguf(const std::vector<std::pair<std::string, std::pair<char, std::string>>>& kvs,
    const std::vector<TSpec>& tensors) {
    // kvs: {key, {type: 'u'|'f'|'s'|'b', value-as-text}} — u32/f32/str/bool.
    std::vector<std::uint8_t> b;
    put_u32(b, 0x46554747u);  // GGUF
    put_u32(b, 3);            // version
    put_u64(b, tensors.size());
    put_u64(b, kvs.size());
    for (const auto& kv : kvs) {
        put_str(b, kv.first);
        switch (kv.second.first) {
            case 'u': put_u32(b, 4); put_u32(b, std::stoul(kv.second.second)); break;
            case 'b': put_u32(b, 7); b.push_back(kv.second.second == "1" ? 1 : 0); break;
            case 'f': put_u32(b, 6); put_f32(b, std::stof(kv.second.second)); break;
            case 's': put_u32(b, 8); put_str(b, kv.second.second); break;
            default: expect(false, "test bug: kv type");
        }
    }
    std::vector<std::uint64_t> offsets(tensors.size(), 0);
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < tensors.size(); i++) {
        offsets[i] = off;
        off += numel(tensors[i]) * 4;
        off = (off + 31) / 32 * 32;
    }
    for (std::size_t i = 0; i < tensors.size(); i++) {
        put_str(b, tensors[i].name);
        put_u32(b, static_cast<std::uint32_t>(tensors[i].dims.size()));
        for (std::size_t d = tensors[i].dims.size(); d-- > 0;)  // ggml: fastest first
            put_u64(b, tensors[i].dims[d]);
        put_u32(b, 0);  // F32
        put_u64(b, offsets[i]);
    }
    const std::size_t data_start = (b.size() + 31) / 32 * 32;
    b.resize(data_start, 0);
    for (std::size_t i = 0; i < tensors.size(); i++) {
        const std::size_t pos = data_start + static_cast<std::size_t>(offsets[i]);
        const std::size_t bytes = numel(tensors[i]) * 4;
        b.resize(pos + bytes, 0);
        std::vector<std::uint8_t> tmp;
        tmp.reserve(bytes);
        for (std::size_t j = 0; j < numel(tensors[i]); j++) put_f32(tmp, pattern(i, j));
        std::memcpy(b.data() + pos, tmp.data(), bytes);  // values land at their offset
    }
    return std::string(b.begin(), b.end());
}

struct CfgPair {
    std::string key, val;
};

std::string build_dfw1(const std::vector<CfgPair>& cfg, const std::vector<TSpec>& tensors) {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x31574644u);  // DFW1
    put_u32(b, 2);
    put_u32(b, cfg.size());
    for (const CfgPair& p : cfg) {
        put_u32(b, static_cast<std::uint32_t>(p.key.size()));
        b.insert(b.end(), p.key.begin(), p.key.end());
        put_u32(b, static_cast<std::uint32_t>(p.val.size()));
        b.insert(b.end(), p.val.begin(), p.val.end());
    }
    put_u32(b, tensors.size());
    for (std::size_t i = 0; i < tensors.size(); i++) {
        put_u32(b, static_cast<std::uint32_t>(tensors[i].name.size()));
        b.insert(b.end(), tensors[i].name.begin(), tensors[i].name.end());
        put_u32(b, static_cast<std::uint32_t>(tensors[i].dims.size()));
        for (std::uint32_t d : tensors[i].dims) put_u32(b, d);
        put_u64(b, numel(tensors[i]));
        for (std::size_t j = 0; j < numel(tensors[i]); j++) put_f32(b, pattern(i, j));
    }
    return std::string(b.begin(), b.end());
}

std::string write_temp(const std::string& content) {
    static int counter = 0;
    const std::string path = "/tmp/diar_sortformer_w_test_" + std::to_string(counter++) + ".bin";
    FILE* fp = std::fopen(path.c_str(), "wb");
    expect(fp != nullptr, "open temp file");
    std::fwrite(content.data(), 1, content.size(), fp);
    std::fclose(fp);
    return path;
}

// ------------------------------------------------------------ tiny fixture
struct Tiny {
    int d_model = 16, enc_layers = 2, enc_heads = 2, d_ff = 24, conv_k = 3;
    int sub_channels = 8, feat_in = 128, pe_max = 64;
    int xf_layers = 2, xf_hidden = 12, xf_inner = 24, xf_heads = 2;
    int spks = 3;

    std::vector<CfgPair> gguf_kvs() const {
        return {{"sortformer.encoder.d_model", "16"},
            {"sortformer.encoder.n_layers", "2"},
            {"sortformer.encoder.n_heads", "2"},
            {"sortformer.encoder.d_ff", "24"},
            {"sortformer.encoder.conv_kernel_size", "3"},
            {"sortformer.encoder.subsampling_conv_channels", "8"},
            {"sortformer.encoder.feat_in", "128"},
            {"sortformer.encoder.pos_emb_max_len", "64"},
            {"sortformer.transformer.n_layers", "2"},
            {"sortformer.transformer.hidden_size", "12"},
            {"sortformer.transformer.inner_size", "24"},
            {"sortformer.transformer.n_heads", "2"},
            {"sortformer.num_speakers", "3"}};
    }
};

// Independent tensor table for the tiny config (the oracle side).
// dims mirror the documented GGUF layout: stem conv2d [C,1,3,3], stem pw
// [C,C,1,1] (converter keeps it 4-D), conformer pw squeezed [2D,D], dw
// [D,1,K] (kept 3-D), linear [out,in], pe [2*pe_max-1, D], fb [F, 257].
std::vector<TSpec> tiny_tensors(bool with_pe) {
    const int D = 16, C = 8, F = 128, K = 3, H = 2, DK = 8, FF = 24;
    const int X = 12, I = 24, SPK = 3;
    const int fbins = 16;  // 128 -> 64 -> 32 -> 16
    std::vector<TSpec> t = {
        {"encoder.pre_encode.conv.0.weight", {C, 1, 3, 3}},
        {"encoder.pre_encode.conv.0.bias", {C}},
        {"encoder.pre_encode.conv.2.weight", {C, 1, 3, 3}},
        {"encoder.pre_encode.conv.2.bias", {C}},
        {"encoder.pre_encode.conv.3.weight", {C, C, 1, 1}},
        {"encoder.pre_encode.conv.3.bias", {C}},
        {"encoder.pre_encode.conv.5.weight", {C, 1, 3, 3}},
        {"encoder.pre_encode.conv.5.bias", {C}},
        {"encoder.pre_encode.conv.6.weight", {C, C, 1, 1}},
        {"encoder.pre_encode.conv.6.bias", {C}},
        {"encoder.pre_encode.out.weight", {D, C * fbins}},
        {"encoder.pre_encode.out.bias", {D}},
    };
    for (int i = 0; i < 2; i++) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        const std::vector<TSpec> layer = {
            {p + "norm_feed_forward1.weight", {D}},
            {p + "norm_feed_forward1.bias", {D}},
            {p + "feed_forward1.linear1.weight", {FF, D}},
            {p + "feed_forward1.linear1.bias", {FF}},
            {p + "feed_forward1.linear2.weight", {D, FF}},
            {p + "feed_forward1.linear2.bias", {D}},
            {p + "norm_self_att.weight", {D}},
            {p + "norm_self_att.bias", {D}},
            {p + "self_attn.linear_q.weight", {D, D}},
            {p + "self_attn.linear_q.bias", {D}},
            {p + "self_attn.linear_k.weight", {D, D}},
            {p + "self_attn.linear_k.bias", {D}},
            {p + "self_attn.linear_v.weight", {D, D}},
            {p + "self_attn.linear_v.bias", {D}},
            {p + "self_attn.linear_pos.weight", {D, D}},
            {p + "self_attn.pos_bias_u", {H, DK}},
            {p + "self_attn.pos_bias_v", {H, DK}},
            {p + "self_attn.linear_out.weight", {D, D}},
            {p + "self_attn.linear_out.bias", {D}},
            {p + "norm_conv.weight", {D}},
            {p + "norm_conv.bias", {D}},
            {p + "conv.pointwise_conv1.weight", {2 * D, D}},
            {p + "conv.pointwise_conv1.bias", {2 * D}},
            {p + "conv.depthwise_conv.weight", {D, 1, K}},
            {p + "conv.depthwise_conv.bias", {D}},
            {p + "conv.batch_norm.weight", {D}},
            {p + "conv.batch_norm.bias", {D}},
            {p + "conv.batch_norm.running_mean", {D}},
            {p + "conv.batch_norm.running_var", {D}},
            {p + "conv.pointwise_conv2.weight", {D, D}},
            {p + "conv.pointwise_conv2.bias", {D}},
            {p + "norm_feed_forward2.weight", {D}},
            {p + "norm_feed_forward2.bias", {D}},
            {p + "feed_forward2.linear1.weight", {FF, D}},
            {p + "feed_forward2.linear1.bias", {FF}},
            {p + "feed_forward2.linear2.weight", {D, FF}},
            {p + "feed_forward2.linear2.bias", {D}},
            {p + "norm_out.weight", {D}},
            {p + "norm_out.bias", {D}},
        };
        t.insert(t.end(), layer.begin(), layer.end());
    }
    t.push_back({"encoder_proj.weight", {X, D}});
    t.push_back({"encoder_proj.bias", {X}});
    for (int i = 0; i < 2; i++) {
        const std::string p = "transformer.layers." + std::to_string(i) + ".";
        const std::vector<TSpec> layer = {
            {p + "first_sub_layer.query_net.weight", {X, X}},
            {p + "first_sub_layer.query_net.bias", {X}},
            {p + "first_sub_layer.key_net.weight", {X, X}},
            {p + "first_sub_layer.key_net.bias", {X}},
            {p + "first_sub_layer.value_net.weight", {X, X}},
            {p + "first_sub_layer.value_net.bias", {X}},
            {p + "first_sub_layer.out_projection.weight", {X, X}},
            {p + "first_sub_layer.out_projection.bias", {X}},
            {p + "layer_norm_1.weight", {X}},
            {p + "layer_norm_1.bias", {X}},
            {p + "second_sub_layer.dense_in.weight", {I, X}},
            {p + "second_sub_layer.dense_in.bias", {I}},
            {p + "second_sub_layer.dense_out.weight", {X, I}},
            {p + "second_sub_layer.dense_out.bias", {X}},
            {p + "layer_norm_2.weight", {X}},
            {p + "layer_norm_2.bias", {X}},
        };
        t.insert(t.end(), layer.begin(), layer.end());
    }
    t.push_back({"head.first_hidden_to_hidden.weight", {X, X}});
    t.push_back({"head.first_hidden_to_hidden.bias", {X}});
    t.push_back({"head.single_hidden_to_spks.weight", {SPK, X}});
    t.push_back({"head.single_hidden_to_spks.bias", {SPK}});
    t.push_back({"preprocessor.fb", {F, 257}});
    if (with_pe) t.push_back({"encoder.pos_enc.pe", {2 * 64 - 1, D}});
    return t;
}

// index of a name inside the fixture table (for value oracles)
std::size_t index_of(const std::vector<TSpec>& t, const std::string& name) {
    for (std::size_t i = 0; i < t.size(); i++)
        if (t[i].name == name) return i;
    expect(false, "fixture table missing " + name);
    return 0;
}

std::vector<std::pair<std::string, std::pair<char, std::string>>> tiny_kv_typed(const Tiny& tiny) {
    std::vector<std::pair<std::string, std::pair<char, std::string>>> kvs;
    for (const CfgPair& p : tiny.gguf_kvs()) kvs.push_back({p.key, {'u', p.val}});
    return kvs;
}

// ------------------------------------------------------------------ tests
void test_gguf_bind_and_config() {
    const Tiny tiny;
    const std::vector<TSpec> tensors = tiny_tensors(/*with_pe=*/true);
    const std::string blob = build_gguf(tiny_kv_typed(tiny), tensors);
    const diar::SortformerWeights w = diar::SortformerWeights::load(write_temp(blob));

    const diar::SortformerConfig& c = w.config();
    expect(c.d_model == 16 && c.encoder_layers == 2 && c.encoder_heads == 2 &&
               c.encoder_d_ff == 24 && c.conv_kernel == 3 && c.subsampling_conv_channels == 8 &&
               c.feat_in == 128 && c.pos_emb_max_len == 64 && c.transformer_layers == 2 &&
               c.transformer_hidden == 12 && c.transformer_inner == 24 &&
               c.transformer_heads == 2 && c.num_speakers == 3,
        "tiny config parsed from KVs");
    // defaults untouched by the fixture KVs
    expect(c.xscaling && c.subsampling_factor == 8, "config defaults preserved");
    expect_near(c.scoring.pred_score_threshold, 0.25, 0.0, "scoring default 1");
    expect_near(c.scoring.weak_boost_rate, 1.5, 0.0, "scoring default 2");
    expect(c.scoring.sil_frames_per_spk == 3, "scoring default 3");

    expect(w.bound_tensor_count() == tensors.size(), "all tensors bound (incl pe)");
    expect(w.pe_table() != nullptr && w.pe_rows() == 127, "pe bound");

    // pointer spot checks against the fixture value pattern
    const auto check = [&](const char* name, const float* p, std::size_t j) {
        const std::size_t ti = index_of(tensors, name);
        expect_near(p[j], pattern(ti, j), 0.0, std::string("value of ") + name);
    };
    check("encoder.pre_encode.conv.0.weight", w.stem().c0_w, 7);
    check("encoder.pre_encode.out.weight", w.stem().out_w, 100);
    check("encoder.layers.1.conv.batch_norm.running_var", w.conformer(1).conv.bn_var, 3);
    check("encoder.layers.0.self_attn.pos_bias_v", w.conformer(0).attn.bv, 5);
    check("encoder.layers.1.feed_forward2.linear2.weight", w.conformer(1).ff2_w2, 31);
    check("encoder_proj.weight", w.proj_w(), 23);
    check("transformer.layers.0.second_sub_layer.dense_out.weight", w.transformer(0).f2_w, 17);
    check("transformer.layers.1.first_sub_layer.key_net.bias", w.transformer(1).k_b, 4);
    check("head.single_hidden_to_spks.weight", w.head_spks_w(), 11);
    check("preprocessor.fb", w.mel_basis(), 257 * 3 + 5);
    check("encoder.pos_enc.pe", w.pe_table(), 16 * 40 + 2);

    // separate projections must be separate tensors (qkv-fusion trap)
    expect(w.conformer(0).attn.q_w != w.conformer(0).attn.k_w &&
               w.conformer(0).attn.k_w != w.conformer(0).attn.v_w &&
               w.conformer(0).attn.v_w != w.conformer(0).attn.pos_w,
        "q/k/v/pos bound to distinct tensors");

    expect_throws_substr("conformer layer 2",
        [&] { (void)w.conformer(2); }, "conformer index checked");
}

void test_dfw1_v2_bind() {
    const std::vector<TSpec> tensors = tiny_tensors(/*with_pe=*/false);
    const std::vector<CfgPair> cfg = {
        {"sortformer.encoder.d_model", "16"},
        {"sortformer.encoder.n_layers", "2"},
        {"sortformer.encoder.n_heads", "2"},
        {"sortformer.encoder.d_ff", "24"},
        {"sortformer.encoder.conv_kernel_size", "3"},
        {"sortformer.encoder.subsampling_conv_channels", "8"},
        {"sortformer.encoder.pos_emb_max_len", "64"},
        {"sortformer.transformer.n_layers", "2"},
        {"sortformer.transformer.hidden_size", "12"},
        {"sortformer.transformer.inner_size", "24"},
        {"sortformer.transformer.n_heads", "2"},
        {"sortformer.num_speakers", "3"},
    };
    const diar::SortformerWeights w =
        diar::SortformerWeights::load(write_temp(build_dfw1(cfg, tensors)));
    expect(w.config().d_model == 16 && w.config().encoder_layers == 2 &&
               w.config().num_speakers == 3,
        "dfw1 v2 config parsed");
    expect(w.config().feat_in == 128, "dfw1 feat_in default");
    expect(w.pe_table() == nullptr, "dfw1 without pe -> formula route");
    expect(w.bound_tensor_count() == tensors.size(), "dfw1 bound count");

    // DFW1 WITH pe present: accepted and bound (documented bonus).
    const std::vector<TSpec> tensors_pe = tiny_tensors(/*with_pe=*/true);
    const diar::SortformerWeights w2 =
        diar::SortformerWeights::load(write_temp(build_dfw1(cfg, tensors_pe)));
    expect(w2.pe_table() != nullptr && w2.pe_rows() == 127, "dfw1 pe accepted");
    expect(w2.bound_tensor_count() == tensors_pe.size(), "dfw1 pe counted");
}

void test_missing_and_extra_loud() {
    const Tiny tiny;
    std::vector<TSpec> tensors = tiny_tensors(true);
    // drop two tensors (one mid-layer, one pe)
    std::vector<TSpec> minus;
    for (const TSpec& t : tensors)
        if (t.name != "encoder.layers.1.conv.batch_norm.running_var" &&
            t.name != "encoder.layers.0.self_attn.linear_pos.weight")
            minus.push_back(t);
    expect_throws_substr("encoder.layers.1.conv.batch_norm.running_var",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), minus))); },
        "missing tensors listed (1)");
    expect_throws_substr("encoder.layers.0.self_attn.linear_pos.weight",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), minus))); },
        "missing tensors listed (2)");

    // pe missing on the GGUF route
    std::vector<TSpec> no_pe = tiny_tensors(false);
    expect_throws_substr("pos_enc.pe",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), no_pe))); },
        "gguf route requires pe");

    // extra unknown tensor
    std::vector<TSpec> extra = tiny_tensors(true);
    extra.push_back({"encoder.layers.0.norm_ff1.weight", {16}});
    expect_throws_substr("norm_ff1",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), extra))); },
        "extra tensor rejected");
}

void test_shape_mismatch_loud() {
    const Tiny tiny;
    std::vector<TSpec> tensors = tiny_tensors(true);
    for (TSpec& t : tensors)
        if (t.name == "encoder.pre_encode.out.weight") t.dims = {16, 9};  // wrong 2nd dim
    expect_throws_substr("encoder.pre_encode.out.weight",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), tensors))); },
        "shape mismatch names tensor");
    // canonicalization: [D,1,K] depthwise is accepted as [D,K] — covered by
    // the happy path; a genuinely wrong depthwise rank is not:
    std::vector<TSpec> tensors2 = tiny_tensors(true);
    for (TSpec& t : tensors2)
        if (t.name == "encoder.layers.0.conv.depthwise_conv.weight") t.dims = {16, 9};
    expect_throws_substr("depthwise_conv.weight",
        [&] { (void)diar::SortformerWeights::load(
                  write_temp(build_gguf(tiny_kv_typed(tiny), tensors2))); },
        "depthwise [D,K] vs [D,1,K] both fine but [D,K] wrong-K caught");
}

void test_config_gates() {
    const Tiny tiny;
    const std::vector<TSpec> tensors = tiny_tensors(true);
    auto load_with = [&](std::vector<std::pair<std::string, std::pair<char, std::string>>> kvs) {
        for (const auto& kv : tiny_kv_typed(tiny)) kvs.push_back(kv);
        return diar::SortformerWeights::load(write_temp(build_gguf(kvs, tensors)));
    };
    expect_throws_substr("conv_norm",
        [&] { (void)load_with({{"sortformer.encoder.conv_norm", {'s', "layer_norm"}}}); },
        "conv_norm gate");
    expect_throws_substr("pre_ln",
        [&] { (void)load_with({{"sortformer.transformer.pre_ln", {'b', "1"}}}); },
        "pre_ln gate");
    expect_throws_substr("use_bias",
        [&] { (void)load_with({{"sortformer.encoder.use_bias", {'b', "0"}}}); },
        "use_bias gate");
    expect_throws_substr("n_fft",
        [&] { (void)load_with({{"sortformer.preprocessor.n_fft", {'u', "1024"}}}); },
        "preprocessor pin gate");
    expect_throws_substr("sample_rate",
        [&] { (void)load_with({{"sortformer.preprocessor.sample_rate", {'u', "8000"}}}); },
        "sample_rate pin gate");
}

void test_default_names_table() {
    const diar::SortformerConfig cfg;  // production defaults
    const std::vector<std::string> names = diar::SortformerWeights::expected_tensor_names(cfg, true);
    expect(names.size() == 971, "default model = 971 tensors (got " + std::to_string(names.size()) + ")");  // 12 stem + 17x39 + 2 proj + 18x16 + 4 head + fb + pe
    const auto has = [&](const char* n) {
        return std::find(names.begin(), names.end(), std::string(n)) != names.end();
    };
    expect(has("encoder.layers.16.self_attn.pos_bias_v"), "last conformer pos_bias_v");
    expect(has("encoder.layers.0.conv.pointwise_conv2.weight"), "conformer conv prefix");
    expect(has("transformer.layers.17.second_sub_layer.dense_out.weight"), "last transformer tensor");
    expect(has("preprocessor.fb") && has("encoder.pos_enc.pe"), "synthesized tensors");
    expect(!has("encoder.layers.0.conv_module.pointwise_conv2.weight"), "no torch conv_module prefix");
    expect(diar::SortformerWeights::expected_tensor_names(cfg, false).size() == 970,
        "dfw1 route drops pe");
}

void test_bad_magic_and_move() {
    static_assert(!std::is_copy_constructible<diar::SortformerWeights>::value,
        "SortformerWeights is move-only");
    expect_throws_substr("unknown container magic",
        [&] { (void)diar::SortformerWeights::load(write_temp("junk!")); },
        "magic sniff");
    // move keeps the views valid (heap buffers move with the vectors)
    const Tiny tiny;
    const std::vector<TSpec> tensors = tiny_tensors(true);
    diar::SortformerWeights w =
        diar::SortformerWeights::load(write_temp(build_gguf(tiny_kv_typed(tiny), tensors)));
    const float* q_before = w.conformer(0).attn.q_w;
    const float expect_val = q_before[3];
    diar::SortformerWeights moved = std::move(w);
    expect(moved.conformer(0).attn.q_w == q_before && moved.conformer(0).attn.q_w[3] == expect_val,
        "views survive move");
}

}  // namespace

int main() {
    test_gguf_bind_and_config();
    test_dfw1_v2_bind();
    test_missing_and_extra_loud();
    test_shape_mismatch_loud();
    test_config_gates();
    test_default_names_table();
    test_bad_magic_and_move();
    std::cout << "PASS: sortformer weight arena tests\n";
    return 0;
}
