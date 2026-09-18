// k5_runner — M2 Stage 3.3 free-running engine driver for the Kaggle kernel.
//
// Modes:
//   --run --weights W --audio A.f32 --out PREFIX [--feed whole|drip]
//         [--offline]
//       Loads weights (GGUF or DFW1, magic-detected by SortformerWeights),
//       feeds raw little-endian f32 16k mono PCM from A.f32 through the
//       DiarEngine free-running, and writes next to PREFIX:
//         PREFIX.meta.json     run/config echo + n_frames + chunk count
//         PREFIX.pregate.f32   AOSC-emitted chain, n_frames x n_spk
//         PREFIX.postgate.f32  BirthGate timeline,  n_frames x n_spk
//         PREFIX.ledger.json   per-chunk ChunkLedgerEntry rows
//         PREFIX.aosc.json     per-chunk AOSC snapshot index (after update)
//         PREFIX.aosc.f32      snapshot payloads in index order
//   --full-offline --weights W --audio A.f32 --out PREFIX
//       Upstream DiarModel::diarize_offline: peak-normalize, whole-file FE,
//       one run_chunk with empty state — raw probs, NO BirthGate (mirrors
//       the production `--offline` flag; pregate/postgate written identical).
//   --probe-weights --weights W
//       Full load (loud on any coverage/shape failure) + config echo json.
//   --expected-names
//       Prints the tiny-config expected tensor names as json (pytest pin:
//       the python DFW1 name mirror must equal this list exactly).
//   --selftest
//       Writes a tiny DFW1 v2 to /tmp (LCG-random tensors, correct names +
//       shapes for the tiny config), loads it back, runs 1.5 s of synthetic
//       audio in both feed modes, checks output well-formedness, prints
//       "k5-runner selftest ok". Kernel-side infra gate before real weights.
//
// Everything is deterministic (LCG + fixed audio), so a green selftest here
// means the exact binary the kernel compiled can ingest a DFW1 and produce
// parseable outputs — infra, not numerics (numerics are the kernel's gates).
#include "diar/engine.hpp"
#include "diar/gguf.hpp"
#include "diar/sortformer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using AoscSnapshot = diar::DiarEngine::AoscSnapshot;

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "k5_runner: %s\n", msg.c_str());
    std::exit(2);
}

// ---------------------------------------------------------------------------
// Tiny config shared by --selftest and --expected-names. Must stay in sync
// with tests/test_dfw1_pywriter.py (the pytest pins name equality, so any
// drift fails loudly on both sides).
// ---------------------------------------------------------------------------
struct TinyCfg {
    int d_model = 16;
    int enc_layers = 2;
    int heads = 4;
    int d_ff = 32;
    int kernel = 9;
    int sub = 8;
    int sub_ch = 8;
    int feat_in = 128;  // FE-pinned: parse_config rejects any other n_mels
    int pe_max = 512;
    int tf_layers = 2;
    int tf_hidden = 16;
    int tf_inner = 32;
    int tf_heads = 4;
    int n_spk = 4;
};
const TinyCfg kTiny{};

int freq_bins(int feat_in, int sub) {
    int stages = 0;
    for (int f = sub; f > 1; f >>= 1) ++stages;
    int len = feat_in;
    for (int s = 0; s < stages; ++s) len = (len - 1) / 2 + 1;
    return len;
}

std::vector<std::string> tiny_expected_names() {
    const TinyCfg& c = kTiny;
    const std::int64_t D = c.d_model, C = c.sub_ch, F = c.feat_in, K = c.kernel,
                      FF = c.d_ff, X = c.tf_hidden, I = c.tf_inner, SPK = c.n_spk;
    const int fbins = freq_bins(c.feat_in, c.sub);
    std::vector<std::string> n;
    auto add = [&](const std::string& s) { n.push_back(s); };
    add("encoder.pre_encode.conv.0.weight");
    add("encoder.pre_encode.conv.0.bias");
    add("encoder.pre_encode.conv.2.weight");
    add("encoder.pre_encode.conv.2.bias");
    add("encoder.pre_encode.conv.3.weight");
    add("encoder.pre_encode.conv.3.bias");
    add("encoder.pre_encode.conv.5.weight");
    add("encoder.pre_encode.conv.5.bias");
    add("encoder.pre_encode.conv.6.weight");
    add("encoder.pre_encode.conv.6.bias");
    add("encoder.pre_encode.out.weight");
    add("encoder.pre_encode.out.bias");
    for (int i = 0; i < c.enc_layers; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        add(p + "norm_feed_forward1.weight");
        add(p + "norm_feed_forward1.bias");
        add(p + "feed_forward1.linear1.weight");
        add(p + "feed_forward1.linear1.bias");
        add(p + "feed_forward1.linear2.weight");
        add(p + "feed_forward1.linear2.bias");
        add(p + "norm_self_att.weight");
        add(p + "norm_self_att.bias");
        add(p + "self_attn.linear_q.weight");
        add(p + "self_attn.linear_q.bias");
        add(p + "self_attn.linear_k.weight");
        add(p + "self_attn.linear_k.bias");
        add(p + "self_attn.linear_v.weight");
        add(p + "self_attn.linear_v.bias");
        add(p + "self_attn.linear_pos.weight");
        add(p + "self_attn.pos_bias_u");
        add(p + "self_attn.pos_bias_v");
        add(p + "self_attn.linear_out.weight");
        add(p + "self_attn.linear_out.bias");
        add(p + "norm_conv.weight");
        add(p + "norm_conv.bias");
        add(p + "conv.pointwise_conv1.weight");
        add(p + "conv.pointwise_conv1.bias");
        add(p + "conv.depthwise_conv.weight");
        add(p + "conv.depthwise_conv.bias");
        add(p + "conv.batch_norm.weight");
        add(p + "conv.batch_norm.bias");
        add(p + "conv.batch_norm.running_mean");
        add(p + "conv.batch_norm.running_var");
        add(p + "conv.pointwise_conv2.weight");
        add(p + "conv.pointwise_conv2.bias");
        add(p + "norm_feed_forward2.weight");
        add(p + "norm_feed_forward2.bias");
        add(p + "feed_forward2.linear1.weight");
        add(p + "feed_forward2.linear1.bias");
        add(p + "feed_forward2.linear2.weight");
        add(p + "feed_forward2.linear2.bias");
        add(p + "norm_out.weight");
        add(p + "norm_out.bias");
    }
    add("encoder_proj.weight");
    add("encoder_proj.bias");
    for (int i = 0; i < c.tf_layers; ++i) {
        const std::string p = "transformer.layers." + std::to_string(i) + ".";
        add(p + "first_sub_layer.query_net.weight");
        add(p + "first_sub_layer.query_net.bias");
        add(p + "first_sub_layer.key_net.weight");
        add(p + "first_sub_layer.key_net.bias");
        add(p + "first_sub_layer.value_net.weight");
        add(p + "first_sub_layer.value_net.bias");
        add(p + "first_sub_layer.out_projection.weight");
        add(p + "first_sub_layer.out_projection.bias");
        add(p + "layer_norm_1.weight");
        add(p + "layer_norm_1.bias");
        add(p + "second_sub_layer.dense_in.weight");
        add(p + "second_sub_layer.dense_in.bias");
        add(p + "second_sub_layer.dense_out.weight");
        add(p + "second_sub_layer.dense_out.bias");
        add(p + "layer_norm_2.weight");
        add(p + "layer_norm_2.bias");
    }
    add("head.first_hidden_to_hidden.weight");
    add("head.first_hidden_to_hidden.bias");
    add("head.single_hidden_to_spks.weight");
    add("head.single_hidden_to_spks.bias");
    add("preprocessor.fb");
    (void)D; (void)C; (void)F; (void)K; (void)FF; (void)X; (void)I; (void)SPK; (void)fbins;
    return n;
}

std::vector<std::vector<std::uint32_t>> tiny_expected_shapes() {
    const TinyCfg& c = kTiny;
    const std::int64_t D = c.d_model, C = c.sub_ch, F = c.feat_in, K = c.kernel,
                      FF = c.d_ff, X = c.tf_hidden, I = c.tf_inner, SPK = c.n_spk;
    const int fbins = freq_bins(c.feat_in, c.sub);
    const auto u = [](std::vector<std::int64_t> v) {
        std::vector<std::uint32_t> r;
        for (std::int64_t x : v) r.push_back(static_cast<std::uint32_t>(x));
        return r;
    };
    std::vector<std::vector<std::uint32_t>> s;
    auto add = [&](std::vector<std::int64_t> v) { s.push_back(u(std::move(v))); };
    // order must match tiny_expected_names() exactly
    add({C, 1, 3, 3}); add({C});                          // conv.0
    add({C, 1, 3, 3}); add({C});                          // conv.2
    add({C, C, 1, 1}); add({C});                          // conv.3
    add({C, 1, 3, 3}); add({C});                          // conv.5
    add({C, C, 1, 1}); add({C});                          // conv.6
    add({D, C * fbins}); add({D});                        // out
    for (int i = 0; i < c.enc_layers; ++i) {
        add({D}); add({D});                               // n_ff1
        add({FF, D}); add({FF}); add({D, FF}); add({D});  // ff1
        add({D}); add({D});                               // n_sa
        add({D, D}); add({D});                            // q
        add({D, D}); add({D});                            // k
        add({D, D}); add({D});                            // v
        add({D, D});                                      // pos (no bias)
        add({c.heads, D / c.heads}); add({c.heads, D / c.heads});  // bias u/v
        add({D, D}); add({D});                            // attn out
        add({D}); add({D});                               // n_conv
        add({2 * D, D}); add({2 * D});                    // pw1
        add({D, 1, K}); add({D});                         // dw
        add({D}); add({D}); add({D}); add({D});           // bn
        add({D, D}); add({D});                            // pw2
        add({D}); add({D});                               // n_ff2
        add({FF, D}); add({FF}); add({D, FF}); add({D});  // ff2
        add({D}); add({D});                               // n_out
    }
    add({X, D}); add({X});                                // proj
    for (int i = 0; i < c.tf_layers; ++i) {
        add({X, X}); add({X}); add({X, X}); add({X});     // q/k
        add({X, X}); add({X}); add({X, X}); add({X});     // v/o
        add({X}); add({X});                               // ln1
        add({I, X}); add({I}); add({X, I}); add({X});     // dense in/out
        add({X}); add({X});                               // ln2
    }
    add({X, X}); add({X});                                // head fc
    add({SPK, X}); add({SPK});                            // head out
    add({F, 257});                                        // fb
    return s;
}

// ---------------------------------------------------------------------------
// DFW1 v2 writer (byte contract: include/diar/gguf.hpp + src/gguf.cpp).
// ---------------------------------------------------------------------------
void put_u32(std::string& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::string& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void put_str(std::string& b, const std::string& s) {
    put_u32(b, static_cast<std::uint32_t>(s.size()));
    b.append(s);
}

struct TinyTensor {
    std::string name;
    std::vector<std::uint32_t> dims;
    std::vector<float> data;
};

// Deterministic LCG in [0,1) -> scaled pseudo-gaussians; degenerate stacks are
// fine here (selftest checks plumbing, not numerics).
float lcg_f(std::uint64_t& st) {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    const std::uint32_t r = static_cast<std::uint32_t>(st >> 33);
    return static_cast<float>(r) / 4294967296.0F * 2.0F - 1.0F;
}

std::string write_tiny_dfw1(const std::string& path) {
    const auto names = tiny_expected_names();
    const auto shapes = tiny_expected_shapes();
    if (names.size() != shapes.size()) die("internal: name/shape count mismatch");
    std::uint64_t st = 0x9E3779B97F4A7C15ull;
    std::string blob;
    blob.append("DFW1", 4);
    put_u32(blob, 2);
    // v2 config section: exactly the keys parse_config reads; values are the
    // decimal text forms (kv_u32/kv_f32/kv_string parse them back).
    const std::vector<std::pair<std::string, std::string>> cfg = {
        {"sortformer.encoder.d_model", "16"},
        {"sortformer.encoder.n_layers", "2"},
        {"sortformer.encoder.n_heads", "4"},
        {"sortformer.encoder.d_ff", "32"},
        {"sortformer.encoder.conv_kernel_size", "9"},
        {"sortformer.encoder.subsampling_factor", "8"},
        {"sortformer.encoder.subsampling_conv_channels", "8"},
        {"sortformer.encoder.feat_in", "128"},
        {"sortformer.encoder.xscaling", "1"},
        {"sortformer.encoder.pos_emb_max_len", "512"},
        {"sortformer.transformer.n_layers", "2"},
        {"sortformer.transformer.hidden_size", "16"},
        {"sortformer.transformer.inner_size", "32"},
        {"sortformer.transformer.n_heads", "4"},
        {"sortformer.num_speakers", "4"},
        {"sortformer.preprocessor.sample_rate", "16000"},
        {"sortformer.preprocessor.window_size", "0.025"},
        {"sortformer.preprocessor.window_stride", "0.01"},
        {"sortformer.preprocessor.n_fft", "512"},
        {"sortformer.preprocessor.features", "128"},
        {"sortformer.preprocessor.preemph", "0.97"},
        {"sortformer.preprocessor.log_zero_guard", std::string("5.9604644775390625e-08")},
    };
    put_u32(blob, static_cast<std::uint32_t>(cfg.size()));
    for (const auto& kv : cfg) {
        put_str(blob, kv.first);
        put_str(blob, kv.second);
    }
    put_u32(blob, static_cast<std::uint32_t>(names.size()));
    for (std::size_t i = 0; i < names.size(); ++i) {
        TinyTensor t;
        t.name = names[i];
        t.dims = shapes[i];
        std::uint64_t count = 1;
        for (std::uint32_t d : t.dims) count *= d;
        t.data.resize(static_cast<std::size_t>(count));
        for (std::size_t j = 0; j < t.data.size(); ++j) t.data[j] = 0.4F * lcg_f(st);
        // keep BN variance positive so the conformer stays finite
        if (t.name.find("running_var") != std::string::npos)
            for (float& v : t.data) v = v * 0.1F + 1.0F;
        // fb multiplies STFT power before log: negative "filterbanks" would
        // feed log() a negative argument -> NaN timeline (dress-rehearsal find)
        if (t.name == "preprocessor.fb")
            for (float& v : t.data) v = std::fabs(v) + 0.1F;
        put_str(blob, t.name);
        put_u32(blob, static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint32_t d : t.dims) put_u32(blob, d);
        put_u64(blob, count);
        const char* p = reinterpret_cast<const char*>(t.data.data());
        blob.append(p, static_cast<std::size_t>(count) * 4);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path);
    f.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    if (!f) die("short write " + path);
    return path;
}

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------
std::string jescape(const std::string& s) {
    std::string r;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            r.push_back('\\');
            r.push_back(ch);
        } else if (static_cast<unsigned char>(ch) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
            r += buf;
        } else {
            r.push_back(ch);
        }
    }
    return r;
}

void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path);
    if (!v.empty())
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * 4));
    if (!f) die("short write " + path);
}

void write_text(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f) die("cannot write " + path);
    f << s;
    if (!f) die("short write " + path);
}

int run_full_offline(const std::string& weights, const std::string& audio,
                     const std::string& out) {
    diar::SortformerWeights w = diar::SortformerWeights::load(weights);
    const diar::SortformerConfig& c = w.config();

    std::vector<float> pcm;
    {
        std::ifstream f(audio, std::ios::binary);
        if (!f) die("cannot open audio " + audio);
        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes < 0 || bytes % 4 != 0) die("audio file size not a f32 multiple");
        pcm.resize(static_cast<std::size_t>(bytes / 4));
        if (!pcm.empty())
            f.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(bytes));
        if (!f) die("short audio read");
    }

    diar::OfflineDiarizationResult r;
    try {
        r = diar::diarize_offline(w, pcm.data(), pcm.size());
    } catch (const std::exception& e) {
        die(std::string("full-offline run failed: ") + e.what());
    }

    // Upstream --offline runs DiarModel::diarize_offline directly: raw frame
    // probs, NO BirthGate (birth_gate_ lives only in the streaming pipeline).
    // We emit both prefixes identically so gate scripts can read either.
    std::string meta = "{\n";
    meta += "  \"weights\": \"" + jescape(weights) + "\",\n";
    meta += "  \"audio\": \"" + jescape(audio) + "\",\n";
    meta += "  \"mode\": \"full-offline\",\n";
    meta += "  \"samples\": " + std::to_string(pcm.size()) + ",\n";
    meta += "  \"n_frames\": " + std::to_string(r.n_frames) + ",\n";
    meta += "  \"n_chunks\": 0,\n";
    meta += "  \"config\": {\"d_model\": " + std::to_string(c.d_model) +
            ", \"encoder_layers\": " + std::to_string(c.encoder_layers) +
            ", \"transformer_layers\": " + std::to_string(c.transformer_layers) +
            ", \"num_speakers\": " + std::to_string(c.num_speakers) +
            ", \"feat_in\": " + std::to_string(c.feat_in) + "}\n";
    meta += "}\n";
    write_text(out + ".meta.json", meta);
    write_f32(out + ".pregate.f32", r.preds);
    write_f32(out + ".postgate.f32", r.preds);
    std::cout << "full-offline: n_frames=" << r.n_frames << "\n";
    return 0;
}

int run_mode(const std::string& weights, const std::string& audio, const std::string& out,
             const std::string& feed, bool offline) {
    diar::SortformerWeights w = diar::SortformerWeights::load(weights);
    const diar::SortformerConfig& c = w.config();

    std::vector<float> pcm;
    {
        std::ifstream f(audio, std::ios::binary);
        if (!f) die("cannot open audio " + audio);
        f.seekg(0, std::ios::end);
        const std::streamoff bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        if (bytes < 0 || bytes % 4 != 0) die("audio file size not a f32 multiple");
        pcm.resize(static_cast<std::size_t>(bytes / 4));
        if (!pcm.empty())
            f.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(bytes));
        if (!f) die("short audio read");
    }

    std::vector<AoscSnapshot> snaps;
    diar::EngineConfig ec;
    if (offline) ec.geometry = diar::StreamGeometry::offline_preset();

    std::string err;
    diar::DiarEngine eng = [&]() {
        try {
            return diar::DiarEngine(std::move(w), ec);
        } catch (const std::exception& e) {
            err = e.what();
            throw;
        }
    }();
    if (!err.empty()) die(err);  // unreachable; keeps -Wunused warning away
    eng.set_aosc_recorder(&snaps);

    try {
        if (feed == "whole") {
            eng.feed_audio(pcm.data(), pcm.size());
        } else if (feed == "drip") {
            for (float s : pcm) eng.feed_audio(&s, 1);
        } else {
            die("unknown --feed mode: " + feed);
        }
        eng.finish();
    } catch (const std::exception& e) {
        die(std::string("engine run failed: ") + e.what());
    }

    const std::int64_t nf = eng.n_frames();
    const std::vector<diar::ChunkLedgerEntry>& led = eng.chunk_ledger();

    std::string meta = "{\n";
    meta += "  \"weights\": \"" + jescape(weights) + "\",\n";
    meta += "  \"audio\": \"" + jescape(audio) + "\",\n";
    meta += "  \"feed\": \"" + jescape(feed) + "\",\n";
    meta += "  \"offline\": " + std::string(offline ? "true" : "false") + ",\n";
    meta += "  \"samples\": " + std::to_string(pcm.size()) + ",\n";
    meta += "  \"n_frames\": " + std::to_string(nf) + ",\n";
    meta += "  \"n_chunks\": " + std::to_string(led.size()) + ",\n";
    meta += "  \"n_snapshots\": " + std::to_string(snaps.size()) + ",\n";
    meta += "  \"config\": {\"d_model\": " + std::to_string(c.d_model) +
            ", \"encoder_layers\": " + std::to_string(c.encoder_layers) +
            ", \"transformer_layers\": " + std::to_string(c.transformer_layers) +
            ", \"num_speakers\": " + std::to_string(c.num_speakers) +
            ", \"feat_in\": " + std::to_string(c.feat_in) + "}\n";
    meta += "}\n";
    write_text(out + ".meta.json", meta);

    {
        std::string s = "[\n";
        for (std::size_t i = 0; i < led.size(); ++i) {
            const diar::ChunkLedgerEntry& e = led[i];
            s += "  {\"chunk\": " + std::to_string(e.chunk_index) +
                 ", \"t_mel\": " + std::to_string(e.t_mel) +
                 ", \"t3\": " + std::to_string(e.t3) +
                 ", \"lc_enc\": " + std::to_string(e.lc_enc) +
                 ", \"rc_enc\": " + std::to_string(e.rc_enc) +
                 ", \"emitted\": " + std::to_string(e.emitted) +
                 ", \"spkcache_frames\": " + std::to_string(e.spkcache_frames) +
                 ", \"fifo_frames\": " + std::to_string(e.fifo_frames) +
                 ", \"window_frames\": " + std::to_string(e.window_frames) +
                 ", \"tail_feat_len\": " + std::to_string(e.tail_feat_len) +
                 "}" +
                 (i + 1 < led.size() ? "," : "") + "\n";
        }
        s += "]\n";
        write_text(out + ".ledger.json", s);
    }

    {
        std::string s = "[\n";
        std::uint64_t off = 0;
        for (std::size_t i = 0; i < snaps.size(); ++i) {
            const AoscSnapshot& a = snaps[i];
            s += "  {\"chunk\": " + std::to_string(static_cast<std::int64_t>(i)) +
                 ", \"spk_frames\": " + std::to_string(a.spk_frames) +
                 ", \"fifo_frames\": " + std::to_string(a.fifo_frames) +
                 ", \"silence_frames\": " + std::to_string(a.silence_frames) +
                 ", \"spk_off\": " + std::to_string(off) +
                 ", \"fifo_off\": " + std::to_string(off + a.spkcache.size()) +
                 ", \"mean_off\": " + std::to_string(off + a.spkcache.size() + a.fifo.size()) +
                 ", \"preds_off\": " +
                 std::to_string(off + a.spkcache.size() + a.fifo.size() + a.mean_sil.size()) +
                 "}" + (i + 1 < snaps.size() ? "," : "") + "\n";
            off += a.spkcache.size() + a.fifo.size() + a.mean_sil.size() +
                   a.spkcache_preds.size();
        }
        s += "]\n";
        write_text(out + ".aosc.json", s);
        std::vector<float> blob;
        blob.reserve(static_cast<std::size_t>(off));
        for (const AoscSnapshot& a : snaps) {
            blob.insert(blob.end(), a.spkcache.begin(), a.spkcache.end());
            blob.insert(blob.end(), a.fifo.begin(), a.fifo.end());
            blob.insert(blob.end(), a.mean_sil.begin(), a.mean_sil.end());
            blob.insert(blob.end(), a.spkcache_preds.begin(), a.spkcache_preds.end());
        }
        write_f32(out + ".aosc.f32", blob);
    }

    write_f32(out + ".pregate.f32", eng.pre_gate_probs());
    write_f32(out + ".postgate.f32", eng.post_gate_probs());
    return 0;
}

int probe_weights(const std::string& weights) {
    diar::SortformerWeights w = diar::SortformerWeights::load(weights);  // loud on coverage
    const diar::SortformerConfig& c = w.config();
    std::string s = "{\n  \"ok\": true,\n";
    s += "  \"d_model\": " + std::to_string(c.d_model) + ",\n";
    s += "  \"encoder_layers\": " + std::to_string(c.encoder_layers) + ",\n";
    s += "  \"encoder_heads\": " + std::to_string(c.encoder_heads) + ",\n";
    s += "  \"transformer_layers\": " + std::to_string(c.transformer_layers) + ",\n";
    s += "  \"num_speakers\": " + std::to_string(c.num_speakers) + ",\n";
    s += "  \"feat_in\": " + std::to_string(c.feat_in) + ",\n";
    s += "  \"pos_emb_max_len\": " + std::to_string(c.pos_emb_max_len) + "\n}\n";
    write_text("/dev/stdout", s);
    return 0;
}

int expected_names_mode() {
    const auto names = tiny_expected_names();
    std::string s = "[\n";
    for (std::size_t i = 0; i < names.size(); ++i) {
        s += "  \"" + jescape(names[i]) + "\"" + (i + 1 < names.size() ? "," : "") + "\n";
    }
    s += "]\n";
    write_text("/dev/stdout", s);
    return 0;
}

int selftest() {
    const std::string wpath = "/tmp/k5_selftest.dfw1";
    write_tiny_dfw1(wpath);
    diar::SortformerWeights w = diar::SortformerWeights::load(wpath);  // loud on any drift
    std::cout << "selftest: tiny DFW1 loaded, tensors=" << w.bound_tensor_count() << "\n";

    // 1.5 s of deterministic stereo-free sine mix at 16k.
    std::vector<float> pcm;
    pcm.reserve(24000);
    for (int i = 0; i < 24000; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        pcm.push_back(static_cast<float>(0.3 * std::sin(2 * M_PI * 220 * t) +
                                         0.2 * std::sin(2 * M_PI * 331 * t + 0.7) +
                                         0.1 * std::sin(2 * M_PI * 517 * t + 1.3)));
    }

    std::vector<AoscSnapshot> snaps;
    diar::EngineConfig ec;
    diar::DiarEngine eng(std::move(w), ec);
    eng.set_aosc_recorder(&snaps);
    eng.feed_audio(pcm.data(), pcm.size());
    eng.finish();
    const std::int64_t nf = eng.n_frames();
    if (nf <= 0) die("selftest: empty timeline");
    if (eng.chunk_ledger().size() != snaps.size())
        die("selftest: ledger/snapshot count mismatch");
    for (std::size_t i = 0; i < snaps.size(); ++i) {
        const auto& a = snaps[i];
        if (static_cast<std::int64_t>(a.spkcache.size()) !=
            static_cast<std::int64_t>(a.spk_frames) * kTiny.d_model)
            die("selftest: spkcache payload size mismatch at chunk " + std::to_string(i));
    }
    if (eng.pre_gate_probs().size() != eng.post_gate_probs().size())
        die("selftest: pre/post gate size mismatch");
    for (float v : eng.pre_gate_probs())
        if (!std::isfinite(v)) die("selftest: non-finite prob in the timeline");
    for (const auto& a : snaps) {
        for (float v : a.spkcache)
            if (!std::isfinite(v)) die("selftest: non-finite spkcache embedding");
        for (float v : a.mean_sil)
            if (!std::isfinite(v)) die("selftest: non-finite mean_sil embedding");
    }

    // Drip-feed determinism: 1-sample granularity must be bit-identical.
    diar::SortformerWeights w2 = diar::SortformerWeights::load(wpath);
    diar::DiarEngine eng2(std::move(w2), diar::EngineConfig{});
    for (float s : pcm) eng2.feed_audio(&s, 1);
    eng2.finish();
    if (eng2.n_frames() != nf) die("selftest: drip feed changed n_frames");
    if (std::memcmp(eng2.pre_gate_probs().data(), eng.pre_gate_probs().data(),
                    eng.pre_gate_probs().size() * 4) != 0)
        die("selftest: drip feed not bit-identical");

    std::cout << "selftest: n_frames=" << nf << " chunks=" << eng.chunk_ledger().size() << "\n";
    std::cout << "k5-runner selftest ok\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode, weights, audio, out, feed = "whole";
    bool offline = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--run") mode = "run";
        else if (a == "--full-offline") mode = "full-offline";
        else if (a == "--probe-weights") mode = "probe";
        else if (a == "--expected-names") mode = "names";
        else if (a == "--selftest") mode = "selftest";
        else if (a == "--weights") weights = next("--weights");
        else if (a == "--audio") audio = next("--audio");
        else if (a == "--out") out = next("--out");
        else if (a == "--feed") feed = next("--feed");
        else if (a == "--offline") offline = true;
        else die("unknown arg: " + a);
    }
    if (mode == "selftest") return selftest();
    if (mode == "names") return expected_names_mode();
    if (mode == "probe") {
        if (weights.empty()) die("--probe-weights needs --weights");
        return probe_weights(weights);
    }
    if (mode == "run") {
        if (weights.empty() || audio.empty() || out.empty())
            die("--run needs --weights, --audio, --out");
        return run_mode(weights, audio, out, feed, offline);
    }
    if (mode == "full-offline") {
        if (weights.empty() || audio.empty() || out.empty())
            die("--full-offline needs --weights, --audio, --out");
        return run_full_offline(weights, audio, out);
    }
    die("no mode: pass --run | --full-offline | --probe-weights | --expected-names | --selftest");
}
