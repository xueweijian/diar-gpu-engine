// Prob-dump oracle: the C++ we inject into upstream app/diarize.cpp,
// compiled standalone HERE (no upstream deps). It must: write the binary
// frame-probability format [int64 LE n_frames][int32 LE n_spk][float32 LE
// row-major frame x spk], return bytes==expected layout, and read back
// bit-identical. If this oracle is green, the injected block's write logic
// is byte-correct by construction (the injection is this code, verbatim).
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// --- BEGIN INJECTED LOGIC (must stay char-identical to the patch) ---
static bool diar_probdump_write(
    const char* path, const float* probs, int64_t n_frames, int n_spk) {
    if (!path || !probs || n_frames <= 0 || n_spk <= 0)
        return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char*>(&n_frames), sizeof(n_frames));
    int32_t spk = static_cast<int32_t>(n_spk);
    out.write(reinterpret_cast<const char*>(&spk), sizeof(spk));
    out.write(
        reinterpret_cast<const char*>(probs),
        static_cast<std::streamsize>(n_frames) * n_spk * sizeof(float));
    out.close();
    return static_cast<bool>(out);
}
// --- END INJECTED LOGIC ---

static int failures = 0;
#define CHECK(cond, msg)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                    \
        }                                                  \
    } while (0)

int main() {
    // 1. roundtrip: 5 frames x 4 spk, edge values (0/1/denormal-ish/neg)
    {
        const int64_t F = 5;
        const int S = 4;
        std::vector<float> probs = {
            0.0f, 1.0f, 0.5f, 0.25f,
            1e-30f, -0.0f, 3.14159f, 2.71828f,
            0.1f, 0.2f, 0.3f, 0.4f,
            0.9f, 0.8f, 0.7f, 0.6f,
            123.456f, -7.5f, 0.0f, 1.0f,
        };
        const std::string p = (std::filesystem::temp_directory_path()
            / "probdump_oracle_case1.f32").string();
        CHECK(diar_probdump_write(p.c_str(), probs.data(), F, S), "write ok");
        std::ifstream in(p, std::ios::binary);
        CHECK(!!in, "reopen ok");
        int64_t rf = -1;
        int32_t rs = -1;
        in.read(reinterpret_cast<char*>(&rf), sizeof(rf));
        in.read(reinterpret_cast<char*>(&rs), sizeof(rs));
        CHECK(rf == F, "n_frames header");
        CHECK(rs == S, "n_spk header");
        std::vector<float> back(F * S, -999.0f);
        in.read(reinterpret_cast<char*>(back.data()), F * S * sizeof(float));
        CHECK(!!in, "payload read ok");
        CHECK(std::memcmp(back.data(), probs.data(), F * S * sizeof(float)) == 0,
              "payload bit-identical");
        // exact byte size: 8 + 4 + F*S*4
        in.seekg(0, std::ios::end);
        CHECK(in.tellg() == static_cast<std::streampos>(8 + 4 + F * S * 4),
              "exact file size");
    }
    // 2. guard rails: null/empty must refuse, never create partial files
    {
        float one = 1.0f;
        CHECK(!diar_probdump_write(nullptr, &one, 1, 1), "null path refused");
        CHECK(!diar_probdump_write("/tmp/x", nullptr, 1, 1), "null probs refused");
        CHECK(!diar_probdump_write("/tmp/x", &one, 0, 1), "zero frames refused");
        CHECK(!diar_probdump_write("/tmp/x", &one, 1, 0), "zero spk refused");
        CHECK(!diar_probdump_write("/tmp/x", &one, -3, 1), "neg frames refused");
    }
    // 3. single frame single speaker (degenerate shape)
    {
        float v = 0.641f;
        const std::string p = (std::filesystem::temp_directory_path()
            / "probdump_oracle_case3.f32").string();
        CHECK(diar_probdump_write(p.c_str(), &v, 1, 1), "1x1 write ok");
        std::ifstream in(p, std::ios::binary);
        int64_t rf = 0;
        int32_t rs = 0;
        float rv = 0;
        in.read(reinterpret_cast<char*>(&rf), sizeof(rf));
        in.read(reinterpret_cast<char*>(&rs), sizeof(rs));
        in.read(reinterpret_cast<char*>(&rv), sizeof(rv));
        CHECK(rf == 1 && rs == 1, "1x1 header");
        CHECK(std::memcmp(&rv, &v, 4) == 0, "1x1 value bit-identical");
    }
    if (failures == 0)
        std::printf("PROBDUMP_ORACLE_PASS\n");
    return failures ? 1 : 0;
}
