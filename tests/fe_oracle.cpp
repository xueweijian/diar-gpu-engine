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

    if (g_failures != 0) {
        std::printf("FE ORACLE: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("FE ORACLE: all cases bit-identical\n");
    return 0;
}
