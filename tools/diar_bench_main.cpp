// diar_bench_main.cpp — M3 Step 6 `diar-bench`: the acceptance deliverable
// (plan §6 + M3-STEP6-ACCEPTANCE-PLAN.md 6c).
//
//   diar-bench --weights <gguf|dfw1> --audio in.f32 [--ref ref.f32]
//              [--ref-face postgate|pregate]
//              [--mode streaming|offline-preset|full-offline]
//              [--route cpu|fp32|fp16] [--reps N] [--label STR]
//              --out PREFIX
//
// Outputs:
//   PREFIX.bench.json    machine fingerprint + route stats + G-B metrics +
//                        G-C determinism + G-D timing (+ per-stage profile
//                        when built with -DDIAR_PROFILE_STAGE)
//   PREFIX.pregate.f32   AOSC chain probs, probdump WIRE format
//   PREFIX.postgate.f32  BirthGate timeline, probdump WIRE format
//                        (12-byte little-endian <qi> header — n_frames i64,
//                        n_spk i32 — + row-major f32 payload; same as the
//                        M1 fixtures, so scripts/ge_cross_diff.py and the
//                        fixture loaders read both)
//
// Route semantics: cpu = the M2 reference engine; fp32 = CudaEncoderRoute
// (Sgemm); fp16 = CudaEncoderRoute (GemmEx 16F-in/32F-acc, Step 5 route,
// opt-in). full-offline always runs the CPU whole-file path (diarize_offline
// has no route hook by design — recorded as route_effective=cpu).
//
// --selftest (CPU build, no files): wire round-trip + metrics oracles so a
// Kaggle round trip never burns on harness bugs (the engine itself is
// pinned by ctest).
#include "diar/diar.hpp"
#include "diar/engine.hpp"
#include "diar/sortformer.hpp"
#include "diar/tailfix.hpp"

#ifdef DIAR_WITH_CUDA
#include "diar/backend_cuda.hpp"
#include "diar/encoder_cuda.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#ifdef DIAR_PROFILE_STAGE
#include "diar/profile.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void die(const std::string& msg) {
    std::cerr << "diar-bench: " << msg << "\n";
    std::exit(1);
}

std::string jescape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            o += '\\';
            o += ch;
        } else {
            o += ch;
        }
    }
    return o;
}

// ---- probdump wire format (12B <qi> header + f32 payload) ----------------

void write_wire_f32(const std::string& path, const float* v, std::int64_t rows,
                    int cols) {
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path);
    const std::int64_t hdr[2] = {rows, cols};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    const std::streamsize n = static_cast<std::streamsize>(rows) * cols * 4;
    if (n > 0)
        f.write(reinterpret_cast<const char*>(v), n);
    if (!f) die("short write " + path);
}

std::vector<float> read_wire_f32(const std::string& path, std::int64_t& rows,
                                 int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    std::int64_t hdr[2] = {0, 0};
    f.read(reinterpret_cast<char*>(hdr), 12);
    if (!f) die("short header read " + path);
    if (hdr[0] < 0 || hdr[1] <= 0) die("bad wire header " + path);
    rows = hdr[0];
    cols = static_cast<int>(hdr[1]);
    std::vector<float> v(static_cast<std::size_t>(rows) * cols);
    if (!v.empty())
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(v.size()) * 4);
    if (!f) die("short payload read " + path);
    return v;
}

// ---- K6 metrics (same formulas as the K6 kernel so numbers are comparable)

struct Metrics {
    std::int64_t rows = 0;
    double max_abs = 0, mean_abs = 0, frame_agreement = 1;
    double row_max_p50 = 0, row_max_p95 = 0, row_max_p99 = 0;
    std::int64_t worst_row = -1;
};

Metrics metrics(const std::vector<float>& ref, const std::vector<float>& got,
                std::int64_t ref_rows, std::int64_t got_rows, int spk) {
    Metrics m;
    const std::int64_t n = std::min(ref_rows, got_rows);
    m.rows = n;
    if (n <= 0) return m;
    std::vector<double> rowmax(static_cast<std::size_t>(n), 0.0);
    double sum = 0;
    std::int64_t agree = 0;
    double worst = -1.0;
    for (std::int64_t i = 0; i < n; i++) {
        double rm = 0;
        for (int c = 0; c < spk; c++) {
            const double d = std::fabs(
                static_cast<double>(ref[static_cast<std::size_t>(i) * spk + c]) -
                static_cast<double>(got[static_cast<std::size_t>(i) * spk + c]));
            if (d > m.max_abs) m.max_abs = d;
            if (d > rm) rm = d;
            sum += d;
            const bool ra = ref[static_cast<std::size_t>(i) * spk + c] > 0.5F;
            const bool ga = got[static_cast<std::size_t>(i) * spk + c] > 0.5F;
            if (ra == ga) agree++;
        }
        rowmax[static_cast<std::size_t>(i)] = rm;
        if (rm > worst) {
            worst = rm;
            m.worst_row = i;
        }
    }
    m.mean_abs = sum / (double)(n * spk);
    m.frame_agreement = (double)agree / (double)(n * spk);
    std::sort(rowmax.begin(), rowmax.end());
    const auto pct = [&](double p) {
        const std::size_t idx = std::min(
            rowmax.size() - 1,
            static_cast<std::size_t>(p / 100.0 * (double)rowmax.size()));
        return rowmax[idx];
    };
    m.row_max_p50 = pct(50);
    m.row_max_p95 = pct(95);
    m.row_max_p99 = pct(99);
    return m;
}

std::vector<float> load_audio_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open audio " + path);
    f.seekg(0, std::ios::end);
    const std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % 4 != 0) die("audio size not an f32 multiple: " + path);
    std::vector<float> pcm(static_cast<std::size_t>(bytes / 4));
    if (!pcm.empty())
        f.read(reinterpret_cast<char*>(pcm.data()),
               static_cast<std::streamsize>(bytes));
    if (!f) die("short audio read " + path);
    return pcm;
}

// Route budget: analytic upper bound on the encoder window L for a geometry
// (engine windowing pin: t_mel <= hop + lc + rc, +31 rows of tailfix pad;
// L = spkcache + fifo + subsampled(t_mel)). Margin on top for cap drift.
int route_max_l(const diar::StreamGeometry& g, int sub) {
    const int t_mel_max = g.chunk_len * sub + g.chunk_left_context * sub +
                          g.chunk_right_context * sub + 31;
    return g.spkcache_len + g.fifo_len +
           diar::sortformer_subsampled_len(t_mel_max, sub) + 32;
}

struct RunOut {
    std::vector<float> pre, post;
    std::int64_t frames = 0;
    std::size_t chunks = 0;
    double wall_ms = 0;
};

int run_selftest() {
    // wire round-trip
    const std::vector<float> probs = {0.25F, 0.75F, 0.5F, 0.5F, 1.0F, 0.0F};
    write_wire_f32("/tmp/diar_bench_selftest.f32", probs.data(), 3, 2);
    std::int64_t rows = 0;
    int cols = 0;
    const std::vector<float> back = read_wire_f32(
        "/tmp/diar_bench_selftest.f32", rows, cols);
    if (rows != 3 || cols != 2 || back != probs) die("selftest: wire round-trip");
    // metrics oracle: identical -> zeros; one flipped cell -> exact numbers
    const Metrics m0 = metrics(probs, probs, 3, 3, 2);
    if (m0.max_abs != 0 || m0.frame_agreement != 1.0) die("selftest: identical metrics");
    std::vector<float> flip = probs;
    flip[1] = 1.0F;  // 0.75 -> 1.0: d=0.25, crosses the 0.5 argmax? no (both >0.5)
    const Metrics m1 = metrics(probs, flip, 3, 3, 2);
    if (std::fabs(m1.max_abs - 0.25) > 1e-12 || m1.worst_row != 0 ||
        std::fabs(m1.mean_abs - 0.25 / 6.0) > 1e-12 || m1.frame_agreement != 1.0)
        die("selftest: metrics oracle");
    flip[4] = 0.0F;  // 1.0 -> 0.0: crosses 0.5 -> 1 disagreement of 6 cells
    const Metrics m2 = metrics(probs, flip, 3, 3, 2);
    if (std::fabs(m2.frame_agreement - (5.0 / 6.0)) > 1e-12)
        die("selftest: frame agreement oracle");
    std::cout << "diar-bench selftest: OK\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string weights, audio, ref, ref_face = "postgate", mode = "streaming",
                                       route = "cpu", label, out;
    int reps = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--selftest") return run_selftest();
        else if (a == "--weights") weights = next();
        else if (a == "--audio") audio = next();
        else if (a == "--ref") ref = next();
        else if (a == "--ref-face") ref_face = next();
        else if (a == "--mode") mode = next();
        else if (a == "--route") route = next();
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--label") label = next();
        else if (a == "--out") out = next();
        else die("unknown arg " + a);
    }
    if (weights.empty() || audio.empty() || out.empty())
        die("need --weights --audio --out (or --selftest)");
    if (reps < 1) die("--reps >= 1");
    if (mode != "streaming" && mode != "offline-preset" && mode != "full-offline")
        die("unknown --mode " + mode);
    if (route != "cpu" && route != "fp32" && route != "fp16")
        die("unknown --route " + route);
#ifndef DIAR_WITH_CUDA
    if (route != "cpu") die("built without DIAR_WITH_CUDA: only --route cpu");
#endif
    if (label.empty()) label = mode + "-" + route;

    const std::vector<float> pcm = load_audio_f32(audio);

    // ---- fingerprint ------------------------------------------------------
    std::string fp_device = "cpu", fp_cc = "", fp_sms = "", fp_driver = "",
                fp_runtime = "", fp_cublas = "";
#ifdef DIAR_WITH_CUDA
    {
        int drv = 0, rt = 0;
        cudaDriverGetVersion(&drv);
        cudaRuntimeGetVersion(&rt);
        fp_driver = std::to_string(drv);
        fp_runtime = std::to_string(rt);
        if (route != "cpu") {
            const int cc = diar::backend::bench_device_cc();
            fp_device = diar::backend::bench_device_name();
            fp_cc = std::to_string(cc / 10) + "." + std::to_string(cc % 10);
            fp_sms = std::to_string(diar::backend::bench_device_sm_count());
            cublasHandle_t h;
            if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
                int v = 0;
                if (cublasGetVersion(h, &v) == CUBLAS_STATUS_SUCCESS)
                    fp_cublas = std::to_string(v);
                cublasDestroy(h);
            }
        }
    }
#endif

    std::string route_effective = route;
    std::string json = "{\n";
    json += "  \"label\": \"" + jescape(label) + "\",\n";
    json += "  \"mode\": \"" + mode + "\",\n";
    json += "  \"route\": \"" + route + "\",\n";
    json += "  \"fingerprint\": {\"device\": \"" + jescape(fp_device) +
            "\", \"cc\": \"" + fp_cc + "\", \"sms\": \"" + fp_sms +
            "\", \"driver\": \"" + fp_driver + "\", \"runtime\": \"" +
            fp_runtime + "\", \"cublas\": \"" + fp_cublas +
            "\", \"sass_targets\": \"sm_60,sm_70,sm_75+ptx60\"},\n";

    // ---- full-offline: CPU whole-file path (no route hook by design) ------
    if (mode == "full-offline") {
        route_effective = "cpu";
        diar::SortformerConfig probe_cfg;
        {
            diar::SortformerWeights w = diar::SortformerWeights::load(weights);
            probe_cfg = w.config();
        }
        const auto t0 = std::chrono::steady_clock::now();
        diar::SortformerWeights w = diar::SortformerWeights::load(weights);
        diar::OfflineDiarizationResult r =
            diar::diarize_offline(w, pcm.data(), pcm.size());
        const auto t1 = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int spk = probe_cfg.num_speakers;
        write_wire_f32(out + ".pregate.f32", r.preds.data(),
                       static_cast<std::int64_t>(r.preds.size()) / spk, spk);
        write_wire_f32(out + ".postgate.f32", r.preds.data(),
                       static_cast<std::int64_t>(r.preds.size()) / spk, spk);
        json += "  \"route_effective\": \"cpu\",\n";
        json += "  \"frames\": " + std::to_string(r.n_frames) + ",\n";
        json += "  \"wall_ms\": " + std::to_string(wall_ms) + "\n}\n";
        { std::ofstream f(out + ".bench.json"); f << json; }
        std::cout << json;
        return 0;
    }

    // ---- streaming / offline-preset ---------------------------------------
    diar::EngineConfig ec;
    ec.geometry = mode == "offline-preset" ? diar::StreamGeometry::offline_preset()
                                           : diar::StreamGeometry::streaming();
    diar::SortformerConfig probe_cfg;
    {
        diar::SortformerWeights w = diar::SortformerWeights::load(weights);
        probe_cfg = w.config();
    }
    const int max_l = route_max_l(ec.geometry, probe_cfg.subsampling_factor);

#ifdef DIAR_WITH_CUDA
    diar::backend::CudaEncoderRoute* cuda_route = nullptr;
    if (route != "cpu")
        cuda_route = new diar::backend::CudaEncoderRoute(
            diar::SortformerWeights::load(weights), route == "fp16", max_l);
#endif

    std::vector<RunOut> runs;
    runs.reserve(static_cast<std::size_t>(reps));
    for (int rep = 0; rep < reps; rep++) {
        RunOut ro;
        diar::SortformerWeights w = diar::SortformerWeights::load(weights);
        diar::DiarEngine eng(std::move(w), ec);
#ifdef DIAR_WITH_CUDA
        if (cuda_route) eng.set_encoder_route(cuda_route);
#endif
        const auto t0 = std::chrono::steady_clock::now();
        eng.feed_audio(pcm.data(), pcm.size());
        eng.finish();
        const auto t1 = std::chrono::steady_clock::now();
        ro.pre = eng.pre_gate_probs();
        ro.post = eng.post_gate_probs();
        ro.frames = eng.n_frames();
        ro.chunks = eng.chunk_ledger().size();
        ro.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        runs.push_back(std::move(ro));
    }

    const RunOut& first = runs.front();
    const int spk = probe_cfg.num_speakers;
    write_wire_f32(out + ".pregate.f32", first.pre.data(), first.frames, spk);
    write_wire_f32(out + ".postgate.f32", first.post.data(), first.frames, spk);

    // G-C: determinism across reps (bit-identical)
    bool determinism_ok = true;
    for (std::size_t r = 1; r < runs.size(); r++)
        if (runs[r].post != first.post || runs[r].pre != first.pre)
            determinism_ok = false;

    // G-D: per-rep ms/chunk + RTF
    std::string per_rep;
    double best_ms_per_chunk = 1e30;
    for (std::size_t r = 0; r < runs.size(); r++) {
        const double mpc = runs[r].chunks ? runs[r].wall_ms / (double)runs[r].chunks
                                          : 0.0;
        best_ms_per_chunk = std::min(best_ms_per_chunk, mpc);
        const double rtf =
            (double)pcm.size() / 16000.0 / (runs[r].wall_ms / 1000.0);
        per_rep += std::string(r ? ", " : "") + "{\"rep\": " + std::to_string(r) +
                   ", \"wall_ms\": " + std::to_string(runs[r].wall_ms) +
                   ", \"ms_per_chunk\": " + std::to_string(mpc) +
                   ", \"rtf\": " + std::to_string(rtf) + "}";
    }

    json += "  \"route_effective\": \"" + route_effective + "\",\n";
    json += "  \"route_max_l\": " + std::to_string(max_l) + ",\n";
#ifdef DIAR_WITH_CUDA
    if (cuda_route) {
        json += "  \"route_calls\": " + std::to_string(cuda_route->calls()) +
                ", \"route_refused\": " + std::to_string(cuda_route->refused()) +
                ",\n";
    }
#endif
    json += "  \"samples\": " + std::to_string(pcm.size()) + ",\n";
    json += "  \"frames\": " + std::to_string(first.frames) + ",\n";
    json += "  \"chunks\": " + std::to_string(first.chunks) + ",\n";
    json += "  \"determinism_bit_identical\": " +
            std::string(determinism_ok ? "true" : "false") + ",\n";
    json += "  \"runs\": [" + per_rep + "],\n";
    json += "  \"ms_per_chunk_best\": " + std::to_string(best_ms_per_chunk) + ",\n";

#ifdef DIAR_PROFILE_STAGE
    json += "  \"profile\": " + diar::profile::report_json() + ",\n";
#endif

    if (!ref.empty()) {
        std::int64_t rr = 0;
        int rc = 0;
        const std::vector<float> refv = read_wire_f32(ref, rr, rc);
        if (rc != spk) die("ref n_spk " + std::to_string(rc) + " != model " +
                           std::to_string(spk));
        const std::vector<float>& gotv =
            ref_face == "pregate" ? first.pre : first.post;
        const Metrics m = metrics(refv, gotv, rr, first.frames, spk);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "  \"vs_ref\": {\"face\": \"%s\", \"rows\": %lld, "
            "\"rows_delta\": %lld, \"max_abs\": %.6g, \"mean_abs\": %.6g, "
            "\"frame_agreement\": %.6f, \"row_max_p50\": %.6g, "
            "\"row_max_p95\": %.6g, \"row_max_p99\": %.6g, \"worst_row\": %lld}",
            ref_face.c_str(), (long long)m.rows,
            (long long)(first.frames - rr), m.max_abs, m.mean_abs,
            m.frame_agreement, m.row_max_p50, m.row_max_p95, m.row_max_p99,
            (long long)m.worst_row);
        json += std::string(buf) + "\n";
    }
    // trailing-comma hygiene: the last appended field may end ",\n"
    if (json.size() >= 2 && json[json.size() - 2] == ',') {
        json.erase(json.size() - 2, 1);  // drop the comma, keep the newline
    }
    json += "}\n";
    { std::ofstream f(out + ".bench.json"); f << json; }
    std::cout << json;

#ifdef DIAR_WITH_CUDA
    delete cuda_route;
#endif
    return 0;
}
