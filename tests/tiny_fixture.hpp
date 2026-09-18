// Shared tiny-fixture writers + tensor table for the Stage 3.2 tests and the
// dump tool. The table is INDEPENDENT of the code under test (hand-derived
// from the documented GGUF schema); expected counts are asserted in the
// tests that consume it. Header-only; include after the expect() helpers of
// the including test (tools define their own fatal helper).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstdint>
#include <filesystem>
#include <cstring>
#include <string>
#include <vector>

// Fatal helper required from the includer (tests: exit(1); tool: same).
inline void tiny_fixture_die(const char* msg) {
    std::fprintf(stderr, "tiny_fixture: %s\n", msg);
    std::exit(1);
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

// value_overrides: replace the pattern fill for the named tensors (used to
// embed formula tables or hand-crafted values).
std::string build_gguf(const std::vector<std::pair<std::string, std::pair<char, std::string>>>& kvs,
    const std::vector<TSpec>& tensors,
    const std::map<std::string, std::vector<float>>* value_overrides = nullptr) {
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
            default: tiny_fixture_die("test bug: kv type");
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
        const std::vector<float>* ov = nullptr;
        if (value_overrides != nullptr) {
            const auto it = value_overrides->find(tensors[i].name);
            if (it != value_overrides->end()) ov = &it->second;
        }
        if (ov != nullptr) {
            if (ov->size() != numel(tensors[i]))
                tiny_fixture_die("override size mismatch");
            for (float v : *ov) put_f32(tmp, v);
        } else {
            for (std::size_t j = 0; j < numel(tensors[i]); j++) put_f32(tmp, pattern(i, j));
        }
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
    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path()
        / ("diar_sortformer_w_test_" + std::to_string(counter++) + ".bin")).string();
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) tiny_fixture_die("open temp file");
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
            {"sortformer.encoder.pos_emb_max_len", std::to_string(pe_max)},
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
std::vector<TSpec> tiny_tensors(bool with_pe, int pe_max = 64) {
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
    if (with_pe)
        t.push_back({"encoder.pos_enc.pe",
            {static_cast<std::uint32_t>(2 * pe_max - 1), static_cast<std::uint32_t>(D)}});
    return t;
}

// index of a name inside the fixture table (for value oracles)
std::size_t index_of(const std::vector<TSpec>& t, const std::string& name) {
    for (std::size_t i = 0; i < t.size(); i++)
        if (t[i].name == name) return i;
    tiny_fixture_die(("fixture table missing " + name).c_str());
    return 0;
}

std::vector<std::pair<std::string, std::pair<char, std::string>>> tiny_kv_typed(const Tiny& tiny) {
    std::vector<std::pair<std::string, std::pair<char, std::string>>> kvs;
    for (const CfgPair& p : tiny.gguf_kvs()) kvs.push_back({p.key, {'u', p.val}});
    return kvs;
}

