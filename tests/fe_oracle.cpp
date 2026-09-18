// Differential oracle for src/fe.cpp (Track B frontend DSP port).
//
// Method (same as the AOSC/BirthGate oracles): the oracle class below is a
// MECHANICAL RESKIN of upstream NeMo-Speech.cpp a5b6953
// src/asr/features/fe.cpp — class/struct renamed, repo include dropped,
// logic lines zero-edited. Stripped, and why that is safe:
//   * FeModule / GpuBatcher / compute_gpu_* — the ggml GPU graph, never
//     built on the diar path (DiarModel passes a null backend manager);
//   * the NVTX timing block — observability only;
//   * the ggml includes — only the stripped code needed them.
// The main() below runs both implementations over a shared deterministic
// corpus and requires bit-identical windows, bases and features.
// Build: g++ src/fe.cpp tests/fe_oracle.cpp -Iinclude

#include "diar/diar.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
// <algorithm> is transitively pulled in by the headers above on libstdc++;
// include it explicitly so the oracle compiles standalone on other stdlibs.
#include <algorithm>

namespace {

// ------------------------- oracle: upstream reskin -------------------------

constexpr float kPi = 3.14159265358979323846f;

static float
hz_to_mel(float f)  // slaney (HTK-like used by NeMo default)
{
    return 1127.0f * std::log(1.0f + f / 700.0f);
}

static float
mel_to_hz(float m) {
    return 700.0f * (std::exp(m / 1127.0f) - 1.0f);
}

// Iterative radix-2 FFT over real and imaginary arrays of equal length.
static void
fft_radix2_inplace(std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(re.size());
    if (n <= 1)
        return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * kPi / len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                const float ur = re[i + k];
                const float ui = im[i + k];
                const float vr = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                const float vi = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                const float nre = cur_re * wre - cur_im * wim;
                const float nim = cur_re * wim + cur_im * wre;
                cur_re = nre;
                cur_im = nim;
            }
        }
    }
}

struct OracleMelSpecConfig {
    int sample_rate = 16000;
    int n_fft = 512;
    int n_mels = 80;
    float window_size = 0.025f;   // seconds
    float window_stride = 0.01f;  // seconds
    float preemph = 0.97f;
    bool normalize_per_feature = true;
    float log_zero_guard = 1.0f / 16777216.0f;  // 2^-24
    float fmin = 0.0f;
    float fmax = 0.0f;  // 0 -> sr/2
    bool stft_center_window = false;
    bool hann_periodic = true;
    bool mask_invalid_frames = false;
};

class OracleFE {
   public:
    explicit OracleFE(const OracleMelSpecConfig& cfg) : cfg_(cfg) {
        if (cfg_.fmax <= 0.0f)
            cfg_.fmax = 0.5f * cfg_.sample_rate;
        int v = cfg_.n_fft;
        while (v > 1) {
            if (v & 1)
                throw std::runtime_error("n_fft must be power of 2 for this FE");
            v >>= 1;
        }
        init_window();
        init_mel_basis();
    }

    int win_length() const { return static_cast<int>(cfg_.window_size * cfg_.sample_rate + 0.5f); }
    int hop_length() const {
        return static_cast<int>(cfg_.window_stride * cfg_.sample_rate + 0.5f);
    }
    int n_fft() const { return cfg_.n_fft; }
    int n_mels() const { return cfg_.n_mels; }
    const std::vector<float>& window() const { return window_; }
    const std::vector<float>& mel_basis() const { return mel_basis_; }

    void
    set_mel_basis(const float* fb, int n_mels, int n_bins) {
        const int expected_bins = cfg_.n_fft / 2 + 1;
        if (n_mels != cfg_.n_mels || n_bins != expected_bins) {
            throw std::runtime_error(
                "MelSpectrogramExtractor::set_mel_basis: shape mismatch - got (" +
                std::to_string(n_mels) + ", " + std::to_string(n_bins) + "), expected (" +
                std::to_string(cfg_.n_mels) + ", " + std::to_string(expected_bins) + ")");
        }
        mel_basis_.assign(fb, fb + (size_t)n_mels * (size_t)n_bins);
    }

    void
    compute_padded(
        const float* audio, size_t n_samples, size_t valid_samples,
        std::vector<float>& features, int& n_frames, bool reflect_left, bool normalize) const {
        if (valid_samples > n_samples)
            throw std::invalid_argument("valid audio length exceeds padded length");
        normalize = normalize && cfg_.normalize_per_feature;

        const int hop = hop_length();
        const int win = win_length();
        const int n_fft_ = cfg_.n_fft;

        std::vector<float> pre(n_samples, 0.0f);
        std::copy_n(audio, valid_samples, pre.data());
        if (cfg_.preemph != 0.0f && valid_samples >= 2) {
            for (size_t i = valid_samples - 1; i > 0; --i) {
                pre[i] = pre[i] - cfg_.preemph * pre[i - 1];
            }
        }
        const float* preemphed = pre.data();

        std::vector<float> padded;
        padded.reserve(n_samples + n_fft_);
        if (reflect_left) {
            for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0f);
        }
        for (size_t i = 0; i < n_samples; i++) padded.push_back(preemphed[i]);
        for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0f);

        n_frames = static_cast<int>((padded.size() - n_fft_) / hop + 1);
        if (n_frames <= 0) {
            features.clear();
            n_frames = 0;
            return;
        }

        const int n_bins = n_fft_ / 2 + 1;
        features.assign(static_cast<size_t>(cfg_.n_mels) * n_frames, 0.0f);

        std::vector<float> re(n_fft_, 0.0f);
        std::vector<float> im(n_fft_, 0.0f);

        for (int f = 0; f < n_frames; f++) {
            const int offset = f * hop;
            std::fill(re.begin(), re.end(), 0.0f);
            std::fill(im.begin(), im.end(), 0.0f);
            const int woff = cfg_.stft_center_window ? (n_fft_ - win) / 2 : 0;
            for (int i = 0; i < win; i++) {
                float sample = padded[offset + woff + i];
                re[woff + i] = sample * window_[i];
            }

            fft_radix2_inplace(re, im);

            for (int m = 0; m < cfg_.n_mels; m++) {
                float acc = 0.0f;
                for (int k = 0; k < n_bins; k++) {
                    const float power = re[k] * re[k] + im[k] * im[k];
                    acc += mel_basis_[m * n_bins + k] * power;
                }
                features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels] =
                    std::log(acc + cfg_.log_zero_guard);
            }
        }

        const int valid = std::min(static_cast<int>(valid_samples) / hop, n_frames);
        if (cfg_.mask_invalid_frames) {
            for (int f = valid; f < n_frames; ++f) {
                std::fill_n(features.data() + static_cast<size_t>(f) * cfg_.n_mels,
                            cfg_.n_mels, 0.0f);
            }
        }

        if (!normalize)
            return;

        for (int m = 0; m < cfg_.n_mels; m++) {
            double sum = 0.0;
            for (int f = 0; f < valid; f++)
                sum += features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels];
            const double mean = valid > 0 ? sum / valid : 0.0;
            double var = 0.0;
            for (int f = 0; f < valid; f++) {
                const double d =
                    features[static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels] -
                    mean;
                var += d * d;
            }
            if (valid > 1)
                var /= valid - 1;
            const double inv_std = 1.0 / (std::sqrt(var) + 1e-5);
            for (int f = 0; f < n_frames; f++) {
                const size_t idx = static_cast<size_t>(m) + static_cast<size_t>(f) * cfg_.n_mels;
                features[idx] = f < valid ? static_cast<float>((features[idx] - mean) * inv_std)
                                          : 0.0f;
            }
        }
    }

   private:
    void
    init_window() {
        const int win = win_length();
        window_.resize(win);
        const float denom = cfg_.hann_periodic ? static_cast<float>(win)
                                               : static_cast<float>(win - 1);
        for (int i = 0; i < win; i++) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * i / denom));
        }
    }

    void
    init_mel_basis() {
        const int n_bins = cfg_.n_fft / 2 + 1;
        const int n_mels = cfg_.n_mels;
        mel_basis_.assign(n_mels * n_bins, 0.0f);

        const float mel_min = hz_to_mel(cfg_.fmin);
        const float mel_max = hz_to_mel(cfg_.fmax);
        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            const float m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
            mel_points[i] = mel_to_hz(m);
        }
        std::vector<float> bin_freqs(n_bins);
        for (int i = 0; i < n_bins; i++) {
            bin_freqs[i] = static_cast<float>(i) * cfg_.sample_rate / cfg_.n_fft;
        }
        for (int m = 0; m < n_mels; m++) {
            const float f_left = mel_points[m];
            const float f_center = mel_points[m + 1];
            const float f_right = mel_points[m + 2];
            for (int k = 0; k < n_bins; k++) {
                const float f = bin_freqs[k];
                float w = 0.0f;
                if (f >= f_left && f <= f_center)
                    w = (f - f_left) / (f_center - f_left + 1e-12f);
                else if (f >= f_center && f <= f_right)
                    w = (f_right - f) / (f_right - f_center + 1e-12f);
                mel_basis_[m * n_bins + k] = std::max(0.0f, w);
            }
        }
    }

    OracleMelSpecConfig cfg_;
    std::vector<float> window_;
    std::vector<float> mel_basis_;
};

// ------------------------------- diff harness ------------------------------

int g_failures = 0;

void report(const std::string& case_name, const std::vector<float>& mine,
            const std::vector<float>& ref, int mine_frames, int ref_frames) {
    bool ok = mine_frames == ref_frames && mine.size() == ref.size() &&
              std::memcmp(mine.data(), ref.data(),
                          std::min(mine.size(), ref.size()) * sizeof(float)) == 0;
    if (mine.size() != ref.size()) ok = false;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s: frames %d vs %d, size %zu vs %zu", case_name.c_str(), mine_frames,
                    ref_frames, mine.size(), ref.size());
        if (!mine.empty() && !ref.empty() && mine.size() == ref.size()) {
            std::size_t idx = 0;
            float worst = 0.0f;
            for (std::size_t i = 0; i < mine.size(); ++i) {
                const float d = std::fabs(mine[i] - ref[i]);
                if (d > worst) {
                    worst = d;
                    idx = i;
                }
            }
            std::printf(" (first worst at %zu: %.9g vs %.9g, diff %.3g)", idx, mine[idx], ref[idx],
                        worst);
        }
        std::printf("\n");
    }
}

std::vector<float> gen_audio(unsigned seed, std::size_t n) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

struct CaseConfig {
    const char* name;
    bool diar_style;  // symmetric hann + centered + normalize off + n_mels 128
    bool periodic_hann;
    bool center_window;
    bool normalize_per_feature;
    bool mask_invalid_frames;
    bool serialized_basis;  // random basis via set_mel_basis, else Slaney fallback
    bool custom_mel_range;  // fmin=80, fmax=7600
    float preemph;
};

void run_case(const CaseConfig& c) {
    diar::FrontendConfig mine;
    OracleMelSpecConfig ref;
    const int n_mels = c.diar_style ? 128 : 80;
    mine.n_mels = n_mels;
    ref.n_mels = n_mels;
    mine.hann_periodic = c.periodic_hann;
    ref.hann_periodic = c.periodic_hann;
    mine.center_window = c.center_window;
    ref.stft_center_window = c.center_window;
    mine.normalize_per_feature = c.normalize_per_feature;
    ref.normalize_per_feature = c.normalize_per_feature;
    mine.mask_invalid_frames = c.mask_invalid_frames;
    ref.mask_invalid_frames = c.mask_invalid_frames;
    mine.preemph = c.preemph;
    ref.preemph = c.preemph;
    if (c.custom_mel_range) {
        mine.fmin = 80.0F;
        ref.fmin = 80.0f;
        mine.fmax = 7600.0F;
        ref.fmax = 7600.0f;
    }

    diar::MelSpectrogramExtractor fe_mine(mine);
    OracleFE fe_ref(ref);

    // Window + fallback basis parity right after construction.
    if (std::memcmp(fe_mine.window().data(), fe_ref.window().data(),
                    fe_ref.window().size() * sizeof(float)) != 0 ||
        fe_mine.window().size() != fe_ref.window().size()) {
        ++g_failures;
        std::printf("FAIL %s: window mismatch after init\n", c.name);
    }
    if (!c.serialized_basis &&
        (fe_mine.mel_basis().size() != fe_ref.mel_basis().size() ||
         std::memcmp(fe_mine.mel_basis().data(), fe_ref.mel_basis().data(),
                     fe_ref.mel_basis().size() * sizeof(float)) != 0)) {
        ++g_failures;
        std::printf("FAIL %s: fallback mel basis mismatch\n", c.name);
    }
    if (fe_mine.hop_length() != fe_ref.hop_length() ||
        fe_mine.win_length() != fe_ref.win_length()) {
        ++g_failures;
        std::printf("FAIL %s: hop/win length mismatch\n", c.name);
    }

    // Serialized (model GGUF style) basis: deterministic pseudo-random.
    std::vector<float> basis;
    if (c.serialized_basis) {
        const int n_bins = mine.n_fft / 2 + 1;
        std::mt19937 basis_rng(1234);
        std::uniform_real_distribution<float> basis_dist(0.0F, 1.0F);
        basis.resize(static_cast<std::size_t>(n_mels) * n_bins);
        for (auto& v : basis) v = basis_dist(basis_rng);
        fe_mine.set_mel_basis(basis.data(), n_mels, n_bins);
        fe_ref.set_mel_basis(basis.data(), n_mels, n_bins);
        if (std::memcmp(fe_mine.mel_basis().data(), fe_ref.mel_basis().data(),
                        basis.size() * sizeof(float)) != 0) {
            ++g_failures;
            std::printf("FAIL %s: set_mel_basis mismatch\n", c.name);
        }
    }

    // Lengths exercise: pad edges, sub-hop, multi-frame, non-multiple-of-hop,
    // and the padded-tail (valid < n) case. reflect_left=false stays in-domain
    // (n >= n_fft/2) per the port note on the upstream size_t expression.
    struct LengthCase {
        std::size_t n;
        std::size_t valid;
        bool reflect_left;
        bool normalize_arg;
    };
    const std::vector<LengthCase> lengths = {
        {0, 0, true, false},
        {1, 1, true, true},
        {2, 1, true, false},
        {160, 160, true, true},
        {1600, 1600, true, false},
        {16001, 16001, true, true},
        {16001, 9000, true, true},
        {64000, 64000, true, false},
        {2000, 2000, false, true},
        {16001, 9000, false, false},
    };

    unsigned seed = 7u + static_cast<unsigned>(mine.n_fft);
    for (const auto& lc : lengths) {
        const std::vector<float> audio = gen_audio(seed++, lc.n);
        std::vector<float> f_mine, f_ref;
        int n_mine = -1, n_ref = -1;
        fe_mine.compute_padded(audio.data(), lc.n, lc.valid, f_mine, n_mine, lc.reflect_left,
                               lc.normalize_arg);
        fe_ref.compute_padded(audio.data(), lc.n, lc.valid, f_ref, n_ref, lc.reflect_left,
                              lc.normalize_arg);
        report(std::string(c.name) + "/n=" + std::to_string(lc.n) +
                   "/valid=" + std::to_string(lc.valid) +
                   "/reflect=" + std::to_string(lc.reflect_left ? 1 : 0) +
                   "/norm=" + std::to_string(lc.normalize_arg ? 1 : 0),
               f_mine, f_ref, n_mine, n_ref);
    }
}

// ------------------- streaming mel scheduler (fe.cpp:714 reskin) -----------
// Mechanical reskin of upstream produce_new_mel_frames (NeMo-Speech.cpp
// a5b6953 src/asr/features/fe.cpp:714). One deliberate expansion: upstream
// calls fe.compute(...), whose body is exactly compute_padded(n, n) — the
// reskinned OracleFE has no separate compute(), so the call is expanded to
// its definition here and nowhere else.
int oracle_produce_new_mel_frames(OracleFE& fe, const std::vector<float>& audio,
    std::size_t audio_base, std::int64_t i_start, std::vector<float>& out) {
    const int n_mels = fe.n_mels();
    const int hop = fe.hop_length();
    const int n_fft = fe.n_fft();
    out.clear();
    const std::size_t audio_end = audio_base + audio.size();
    const std::int64_t i_max = (audio_end >= static_cast<std::size_t>(n_fft / 2))
                                   ? static_cast<std::int64_t>((audio_end - n_fft / 2) / hop) + 1
                                   : 0;
    if (i_max <= i_start)
        return 0;
    std::vector<float> partial;
    int n_frames = 0;
    std::int64_t first_global;
    if (i_start * hop < n_fft / 2) {
        fe.compute_padded(audio.data(), audio.size(), audio.size(), partial, n_frames,
            /*reflect_left=*/true, /*normalize=*/false);
        first_global = 0;
    } else {
        const std::int64_t off = i_start * hop - n_fft / 2 - static_cast<std::int64_t>(audio_base);
        fe.compute_padded(audio.data() + off, audio.size() - static_cast<std::size_t>(off),
            audio.size() - static_cast<std::size_t>(off), partial, n_frames,
            /*reflect_left=*/false, /*normalize=*/false);
        first_global = i_start;
    }
    const std::int64_t skip = i_start - first_global;
    const std::int64_t take =
        std::min<std::int64_t>(static_cast<std::int64_t>(n_frames) - skip, i_max - i_start);
    if (take <= 0 || skip < 0)
        return 0;
    out.assign(partial.data() + skip * n_mels, partial.data() + (skip + take) * n_mels);
    return static_cast<int>(take);
}

struct MineProduce {
    int operator()(diar::MelSpectrogramExtractor& fe, const std::vector<float>& audio,
        std::size_t audio_base, std::int64_t i_start, std::vector<float>& out) const {
        return diar::produce_new_mel_frames(fe, audio, audio_base, i_start, out);
    }
};

// Drives a scheduler through a full stream: identical shard sizes, identical
// consume hop, identical buffer trim policy (upstream DiarStream::run_one_chunk
// lines), optionally the finish() decay tail. Returns every released frame.
struct StreamTrace {
    std::vector<float> frames;               // frame-major, n_mels per row
    std::vector<std::int64_t> slice_starts;  // first frame of each mid-stream slice
    int calls = 0;                           // produce() calls that returned frames
};

template <class FE, class Produce>
StreamTrace drive_stream(FE& fe, const std::vector<float>& audio,
    const std::vector<std::size_t>& shards, int hop_mel, int lc_mel, int rc_mel,
    bool finish_tail, float preemph, Produce produce) {
    const int n_fft = fe.n_fft();
    const int hop = fe.hop_length();
    std::vector<float> audio_buf;
    std::size_t audio_base = 0;
    std::int64_t produced = 0;
    std::int64_t consumed = 0;
    StreamTrace tr;

    auto ensure = [&] {
        std::vector<float> fresh;
        const int n = produce(fe, audio_buf, audio_base, produced, fresh);
        if (n <= 0)
            return;
        if (produced > 0 && produced * hop >= n_fft / 2)
            tr.slice_starts.push_back(produced);
        tr.frames.insert(tr.frames.end(), fresh.begin(), fresh.end());
        produced += n;
        ++tr.calls;
    };
    auto drain = [&] {
        while (produced >= consumed + hop_mel + rc_mel) {
            const std::int64_t lc = std::min<std::int64_t>(lc_mel, consumed);
            consumed += hop_mel;
            const std::int64_t keep_mel = std::max<std::int64_t>(consumed - lc, 0);
            const std::int64_t keep_sample =
                std::max<std::int64_t>(keep_mel * hop - n_fft / 2, 0);
            if (keep_sample > static_cast<std::int64_t>(audio_base)) {
                audio_buf.erase(audio_buf.begin(), audio_buf.begin() + (keep_sample - audio_base));
                audio_base = static_cast<std::size_t>(keep_sample);
            }
        }
    };

    std::size_t pos = 0;
    for (std::size_t s : shards) {
        const std::size_t take = std::min(s, audio.size() - pos);
        audio_buf.insert(audio_buf.end(), audio.begin() + pos, audio.begin() + pos + take);
        pos += take;
        ensure();
        drain();
    }
    if (pos < audio.size()) {
        audio_buf.insert(audio_buf.end(), audio.begin() + pos, audio.end());
        ensure();
        drain();
    }
    if (finish_tail && (!audio_buf.empty() || audio_base > 0)) {
        float tail = audio_buf.empty() ? 0.0F : audio_buf.back();
        for (int k = 0; k < n_fft / 2; k++) {
            tail *= preemph;
            audio_buf.push_back(tail);
        }
        ensure();
        drain();
    }
    return tr;
}

void report_stream(const std::string& case_name, const StreamTrace& mine, const StreamTrace& ref) {
    bool ok = mine.calls == ref.calls && mine.slice_starts == ref.slice_starts &&
              mine.frames.size() == ref.frames.size() &&
              (mine.frames.empty() ||
                  std::memcmp(mine.frames.data(), ref.frames.data(),
                      mine.frames.size() * sizeof(float)) == 0);
    if (ok)
        return;
    ++g_failures;
    std::printf("FAIL %s: frames %zu vs %zu, calls %d vs %d, slices %zu vs %zu", case_name.c_str(),
        mine.frames.size(), ref.frames.size(), mine.calls, ref.calls, mine.slice_starts.size(),
        ref.slice_starts.size());
    if (mine.frames.size() == ref.frames.size() && !mine.frames.empty()) {
        std::size_t idx = 0;
        float worst = 0.0F;
        for (std::size_t i = 0; i < mine.frames.size(); ++i) {
            const float d = std::fabs(mine.frames[i] - ref.frames[i]);
            if (d > worst) {
                worst = d;
                idx = i;
            }
        }
        std::printf(" (worst %.9g vs %.9g, diff %.3g)", mine.frames[idx], ref.frames[idx], worst);
    }
    std::printf("\n");
}

// Characterization: outside the first frame of each mid-stream slice (the
// upstream preemphasis boundary: a slice starts exactly at its first window's
// left edge, so that sample is not differenced against its predecessor), the
// streaming stream must equal the whole-file compute bit-for-bit.
void characterize_stream(const std::string& case_name, const StreamTrace& tr,
    const std::vector<float>& whole, int whole_frames, int n_mels) {
    const std::int64_t n =
        std::min<std::int64_t>(static_cast<std::int64_t>(tr.frames.size()) / n_mels, whole_frames);
    std::int64_t skipped = 0, compared = 0;
    for (std::int64_t g = 0; g < n; ++g) {
        if (std::binary_search(tr.slice_starts.begin(), tr.slice_starts.end(), g)) {
            ++skipped;
            continue;
        }
        ++compared;
        if (std::memcmp(tr.frames.data() + static_cast<std::size_t>(g) * n_mels,
                whole.data() + static_cast<std::size_t>(g) * n_mels, n_mels * sizeof(float)) != 0) {
            ++g_failures;
            std::printf("FAIL %s: frame %lld differs from the whole-file compute outside a "
                        "slice start\n", case_name.c_str(), static_cast<long long>(g));
            return;
        }
    }
    std::printf("  stream/%s: %lld frames, %lld compared vs whole-file, %lld slice-start "
                "frames excluded\n", case_name.c_str(), static_cast<long long>(n),
        static_cast<long long>(compared), static_cast<long long>(skipped));
}

void run_streaming_case(const std::string& name, const std::vector<std::size_t>& shards,
    int hop_mel, int lc_mel, int rc_mel, bool finish_tail, bool characterize) {
    diar::FrontendConfig mine_cfg;  // diar wiring defaults (n_mels 128, symmetric hann, centered)
    OracleMelSpecConfig ref_cfg;
    ref_cfg.n_mels = 128;
    ref_cfg.hann_periodic = false;
    ref_cfg.stft_center_window = true;
    ref_cfg.normalize_per_feature = false;
    ref_cfg.mask_invalid_frames = false;

    diar::MelSpectrogramExtractor fe_mine(mine_cfg);
    OracleFE fe_ref(ref_cfg);
    const int n_bins = mine_cfg.n_fft / 2 + 1;
    std::vector<float> basis(static_cast<std::size_t>(128) * n_bins);
    std::mt19937 basis_rng(4321);
    std::uniform_real_distribution<float> basis_dist(0.0F, 1.0F);
    for (auto& v : basis) v = basis_dist(basis_rng);
    fe_mine.set_mel_basis(basis.data(), 128, n_bins);
    fe_ref.set_mel_basis(basis.data(), 128, n_bins);

    std::size_t total = 0;
    for (std::size_t s : shards) total += s;
    const std::vector<float> audio = gen_audio(99u + static_cast<unsigned>(total), total);

    const StreamTrace mine = drive_stream(fe_mine, audio, shards, hop_mel, lc_mel, rc_mel,
        finish_tail, mine_cfg.preemph, MineProduce{});
    const StreamTrace ref = drive_stream(
        fe_ref, audio, shards, hop_mel, lc_mel, rc_mel, finish_tail, ref_cfg.preemph,
        oracle_produce_new_mel_frames);
    report_stream(name, mine, ref);

    if (characterize && finish_tail) {
        std::vector<float> whole;
        int whole_frames = 0;
        // Whole-buffer reference over audio + finish()'s decay tail: the tail
        // pre-emphasizes to exact zeros, which is what NeMo's constant right
        // pad looks like after preemphasis. Every streaming frame must equal
        // this reference except the first frame of each mid-stream slice.
        std::vector<float> ext = audio;
        float tail = ext.empty() ? 0.0F : ext.back();
        for (int k = 0; k < mine_cfg.n_fft / 2; k++) {
            tail *= mine_cfg.preemph;
            ext.push_back(tail);
        }
        fe_mine.compute(ext.data(), ext.size(), whole, whole_frames, /*reflect_left=*/true,
            /*normalize=*/false);
        characterize_stream(name, mine, whole, whole_frames, 128);
    }
}

} // namespace

int main() {
    // diar wiring (the load-bearing case): symmetric hann, centered, FE
    // normalize off, serialized basis from the GGUF.
    run_case({"diar_wiring", true, false, true, false, false, true, false, 0.97F});
    // diar wiring + FE normalization + tail masking on (padded-tail paths).
    run_case({"diar_norm_mask", true, false, true, true, true, true, false, 0.97F});
    // legacy ASR style: periodic hann, right-aligned, Slaney fallback, no preemph.
    run_case({"legacy_asr", false, true, false, false, false, false, false, 0.0F});
    // legacy + normalization + custom mel range (fmin/fmax math).
    run_case({"legacy_norm_range", false, true, false, true, false, false, true, 0.97F});

    // offline_peak_normalize: port vs the upstream formula on edge inputs.
    {
        const std::vector<float> cases[] = {
            {},
            {0.5F, -0.25F, 0.5F},
            {-0.9F, -0.1F},           // negative-only: upstream quirk (max ignores sign)
            {0.0F, 0.0F, 0.0F},       // silence: gain = 1/eps
            {1.0F, -1.0F, 0.25F},
        };
        const float eps = 1e-3F;
        for (const auto& x : cases) {
            const std::vector<float> mine = diar::offline_peak_normalize(x.data(), x.size(), eps);
            float peak = x.empty() ? 0.0F : x[0];
            for (std::size_t i = 1; i < x.size(); ++i) peak = std::max(peak, x[i]);
            const float gain = 1.0F / (peak + eps);
            std::vector<float> ref(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) ref[i] = x[i] * gain;
            report("peak_normalize/n=" + std::to_string(x.size()), mine, ref,
                   static_cast<int>(x.size()), static_cast<int>(x.size()));
        }
    }

    // streaming mel scheduler (produce_new_mel_frames): shard patterns ×
    // buffer-trim policies × finish-tail, mine vs the upstream reskin, then a
    // whole-file characterization of where streaming legitimately differs.
    {
        const std::size_t n = 32000;  // 2 s
        // Whole buffer in one feed: single slice, no mid-stream boundaries.
        run_streaming_case("one_shard", {n}, 160, 0, 0, true, true);
        // Hop-aligned shards (10 mel frames each): every feed starts a slice.
        std::vector<std::size_t> hop_shards;
        for (std::size_t i = 0; i < n; i += 1600) hop_shards.push_back(1600);
        run_streaming_case("hop_shards", hop_shards, 160, 0, 0, true, true);
        // Sub-window shards (one mel frame per feed) + left-context retention.
        std::vector<std::size_t> tiny_shards;
        for (std::size_t i = 0; i < n; i += 160) tiny_shards.push_back(160);
        run_streaming_case("frame_shards", tiny_shards, 160, 0, 0, true, false);
        run_streaming_case("frame_shards_lc", tiny_shards, 160, 80, 0, true, false);
        // 1-sample shards at the start (the reflect-left path re-entry), then
        // large feeds; right context exercised too.
        std::vector<std::size_t> ragged;
        for (int i = 0; i < 700; i++) ragged.push_back(1);
        ragged.push_back(4000);
        ragged.push_back(777);
        ragged.push_back(9000);
        ragged.push_back(37);
        ragged.push_back(40000);  // capped to the remaining audio
        run_streaming_case("ragged_shards", ragged, 160, 0, 80, true, true);
        // No finish tail: the frames NeMo's pad holds back stay unreleased.
        run_streaming_case("no_finish_tail", hop_shards, 160, 0, 0, false, false);
    }

    if (g_failures != 0) {
        std::printf("FE ORACLE: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("FE ORACLE: all cases bit-identical\n");
    return 0;
}
