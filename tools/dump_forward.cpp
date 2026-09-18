// M2 Stage 3.2b — deterministic forward dump for the numpy mirror comparison.
//
// Generates the shared tiny fixture (tests/tiny_fixture.hpp — the SAME
// hand-derived table the C++ tests use), runs sortformer_run_chunk with a
// tap recorder, and writes everything as raw .f32 files plus a flat
// manifest that reference/mirror_sortformer.py consumes. No JSON/zip —
// the format is line-based on purpose:
//   config <key> <value-text>              (same keys as GGUF sortformer.*)
//   input <which> <rows> <cols> <file>
//   tensor <name> <d0> [<d1> ...] <file>   (row-major, outermost first)
//   tap <name> <rows> <cols> <file>
//   END
//
// Usage: dump_forward --out DIR [--t-mel N] [--feat-len N] [--spkcache N] [--fifo N]
// Defaults: t_mel=86 feat_len=80 (NeMo masked tail mode), spkcache=5, fifo=3.

#include "tiny_fixture.hpp"

#include "diar/sortformer.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

void die(const char* msg) {
    std::fprintf(stderr, "dump_forward: %s\n", msg);
    std::exit(1);
}

struct Rec : diar::TapSink {
    std::vector<std::pair<std::string, std::vector<float>>> ordered;
    void tap(const char* name, const float* data, int rows, int cols) override {
        ordered.emplace_back(name,
            std::vector<float>(data, data + static_cast<std::size_t>(rows) * cols));
    }
};

std::vector<float> lcg_block(std::size_t n, unsigned seed, float lo, float hi) {
    std::vector<float> v(n);
    unsigned lcg = seed;
    for (std::size_t i = 0; i < n; i++) {
        lcg = lcg * 1103515245u + 12345u;
        v[i] = lo + (hi - lo) * static_cast<float>((lcg >> 8) & 0xFFFF) / 65535.0F;
    }
    return v;
}

std::vector<float> state_block(int frames, int d, float base) {
    std::vector<float> v(static_cast<std::size_t>(frames) * d);
    for (std::size_t i = 0; i < v.size(); i++) v[i] = base + 0.001F * (i % 53);
    return v;
}

void write_f32(const std::string& path, const std::vector<float>& v) {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) die("open output file");
    if (!v.empty()) std::fwrite(v.data(), 4, v.size(), fp);
    std::fclose(fp);
}

std::string sanitize(std::string f) {
    for (char& ch : f)
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '.'))
            ch = '_';
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "/tmp/stage32_dump";
    int t_mel = 86, feat_len = 80, n_sc = 5, n_fifo = 3;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&i, &argv, argc](void) { return (i + 1 < argc) ? std::atoi(argv[++i]) : -1; };
        if (a == "--out") out = argv[++i];
        else if (a == "--t-mel") t_mel = next();
        else if (a == "--feat-len") feat_len = next();
        else if (a == "--spkcache") n_sc = next();
        else if (a == "--fifo") n_fifo = next();
        else die("unknown arg");
    }

    const Tiny tiny;
    const std::vector<TSpec> tensors = tiny_tensors(true);
    // Same scale rationale as tests/test_sortformer_forward.cpp load_tiny().
    std::map<std::string, std::vector<float>> scaled;
    for (std::size_t i = 0; i < tensors.size(); i++) {
        std::vector<float> v(numel(tensors[i]));
        for (std::size_t j = 0; j < v.size(); j++) v[j] = pattern(i, j) * 0.003F;
        scaled.emplace(tensors[i].name, std::move(v));
    }
    const diar::SortformerWeights w =
        diar::SortformerWeights::load(write_temp(build_gguf(tiny_kv_typed(tiny), tensors, &scaled)));

    const int D = w.config().d_model;
    const std::vector<float> mel = lcg_block(static_cast<std::size_t>(t_mel) * 128, 7u, -0.25F, 0.25F);
    const std::vector<float> sc = state_block(n_sc, D, 0.05F);
    const std::vector<float> fifo = state_block(n_fifo, D, -0.03F);

    Rec rec;
    const diar::SortformerChunkOutput o = diar::sortformer_run_chunk(mel.data(), t_mel, feat_len,
        n_sc > 0 ? sc.data() : nullptr, n_sc, n_fifo > 0 ? fifo.data() : nullptr, n_fifo, w, &rec);

    if (std::system(("mkdir -p " + out).c_str()) != 0) die("mkdir");
    std::string manifest;
    const auto emit = [&manifest, &out](const std::string& kind, const std::string& name,
                          const std::vector<std::uint32_t>& dims, const std::vector<float>& v) {
        const std::string file = sanitize(kind + "_" + name + ".f32");
        write_f32(out + "/" + file, v);
        manifest += kind + " " + name;
        for (std::uint32_t d : dims) manifest += " " + std::to_string(d);
        manifest += " " + file + "\n";
    };

    for (const CfgPair& kv : tiny.gguf_kvs())
        manifest += "config " + kv.key + " " + kv.val + "\n";
    manifest += "config __feat_len " + std::to_string(feat_len) + "\n";
    emit("input", "mel", {static_cast<std::uint32_t>(t_mel), 128}, mel);
    emit("input", "spkcache", {static_cast<std::uint32_t>(n_sc), static_cast<std::uint32_t>(D)}, sc);
    emit("input", "fifo", {static_cast<std::uint32_t>(n_fifo), static_cast<std::uint32_t>(D)}, fifo);
    for (const TSpec& t : tensors) emit("tensor", t.name, t.dims, scaled.at(t.name));
    for (const auto& tp : rec.ordered) {
        const std::string n = tp.first;
        const int L = n_sc + n_fifo + o.chunk_frames;
        int rows = 0, cols = 0;
        if (n == "stem.out") { rows = o.chunk_frames; cols = D; }
        else if (n == "concat.raw" || n == "xscaled") { rows = L; cols = D; }
        else if (n == "pos_emb") { rows = 2 * L - 1; cols = D; }
        else if (n == "proj.out") { rows = L; cols = w.config().transformer_hidden; }
        else if (n == "preds") { rows = L; cols = w.config().num_speakers; }
        else if (n.rfind("conformer.", 0) == 0 || n.rfind("transformer.", 0) == 0) {
            rows = L; cols = n.rfind("conformer.", 0) == 0 ? D : w.config().transformer_hidden;
        } else die("unknown tap");
        emit("tap", n, {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols)}, tp.second);
    }
    manifest += "END\n";

    std::FILE* fp = std::fopen((out + "/manifest.txt").c_str(), "w");
    if (fp == nullptr) die("write manifest");
    std::fwrite(manifest.data(), 1, manifest.size(), fp);
    std::fclose(fp);
    std::printf("dumped %zu tensors + %zu taps to %s (L=%d T3=%d)\n", tensors.size(),
        rec.ordered.size(), out.c_str(), o.total_frames, o.chunk_frames);
    return 0;
}
