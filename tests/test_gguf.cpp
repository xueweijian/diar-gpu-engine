// GGUF + DFW1 reader tests.
//
// The GGUF writer below is an independent byte-level reference implementation
// of the format subset we parse (v3 layout, scalar/string/array KV, F32/F16/
// Q8_0 tensors): every reader behavior is checked against hand-packed bytes,
// never against the reader itself. Q8_0 expectations use d*q closed forms
// with exactly representable fp16 scales; F16 edge cases (subnormal/min
// normal/max/inf/nan) use ldexp/oracle constants.
//
// Layout pins that must move WITH the spec if the writer ever changes:
//   Q8_0 block = fp16 scale d + 32 x int8, w[i] = d * qs[i] (34 B/block)
//   ggml ne[0] = fastest dimension; file integers are little-endian.

#include "diar/gguf.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
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

void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(v & 0xFF);
    b.push_back((v >> 8) & 0xFF);
}
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

std::uint16_t float_to_half_bits(float f) {
    // Test-only; arguments are exactly representable (mantissa <= 10 bits).
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    if ((bits & 0x7FFFFFFFu) == 0) return static_cast<std::uint16_t>(sign);
    const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
    const std::uint32_t man = (bits >> 13) & 0x3FFu;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp + 15) << 10) | man);
}

enum KV : std::uint32_t { kU8 = 0, kI16 = 3, kU32 = 4, kF32 = 6, kBool = 7, kStr = 8, kArr = 9 };

struct KvSpec {
    std::string key{};
    std::uint32_t type = 0;
    std::uint32_t u = 0;
    float f = 0.0F;
    std::string s{};
};

void put_kv(std::vector<std::uint8_t>& b, const KvSpec& kv) {
    put_str(b, kv.key);
    put_u32(b, kv.type);
    switch (kv.type) {
        case kU8: b.push_back(static_cast<std::uint8_t>(kv.u & 0xFF)); break;
        case kBool: b.push_back(kv.u ? 1 : 0); break;
        case kI16: put_u16(b, static_cast<std::uint16_t>(static_cast<int16_t>(kv.u))); break;
        case kU32: put_u32(b, kv.u); break;
        case kF32: put_f32(b, kv.f); break;
        case kStr: put_str(b, kv.s); break;
        case kArr:  // f32 x 2
            put_u32(b, kF32);
            put_u64(b, 2);
            put_f32(b, 1.0F);
            put_f32(b, 2.0F);
            break;
        default: expect(false, "test bug: kv type not wired");
    }
}

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> ne;  // ggml order
    std::uint32_t type;
    std::vector<std::uint8_t> bytes;
};

std::string build_gguf(const std::vector<KvSpec>& kvs, const std::vector<TensorSpec>& tensors,
                       std::uint32_t alignment, std::uint32_t version = 3) {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x46554747u);  // "GGUF"
    put_u32(b, version);
    put_u64(b, tensors.size());
    put_u64(b, kvs.size());
    for (const KvSpec& kv : kvs) put_kv(b, kv);
    std::vector<std::uint64_t> offsets(tensors.size(), 0);
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < tensors.size(); i++) {
        offsets[i] = off;
        off = (off + tensors[i].bytes.size() + alignment - 1) / alignment * alignment;
    }
    for (std::size_t i = 0; i < tensors.size(); i++) {
        put_str(b, tensors[i].name);
        put_u32(b, static_cast<std::uint32_t>(tensors[i].ne.size()));
        for (std::uint64_t d : tensors[i].ne) put_u64(b, d);
        put_u32(b, tensors[i].type);
        put_u64(b, offsets[i]);
    }
    const std::size_t data_start =
        static_cast<std::size_t>((b.size() + alignment - 1) / alignment * alignment);
    b.resize(data_start, 0);
    for (std::size_t i = 0; i < tensors.size(); i++) {
        const std::size_t pos = data_start + static_cast<std::size_t>(offsets[i]);
        if (b.size() < pos + tensors[i].bytes.size()) b.resize(pos + tensors[i].bytes.size(), 0);
        std::memcpy(b.data() + pos, tensors[i].bytes.data(), tensors[i].bytes.size());
    }
    return std::string(b.begin(), b.end());
}

std::vector<std::string>& temp_paths() {
    static std::vector<std::string> paths;
    return paths;
}

std::string write_temp(const std::string& content) {
    static int counter = 0;
    const std::string path = "/tmp/diar_gguf_test_" + std::to_string(counter++) + ".bin";
    FILE* fp = std::fopen(path.c_str(), "wb");
    expect(fp != nullptr, "open temp file");
    std::fwrite(content.data(), 1, content.size(), fp);
    std::fclose(fp);
    temp_paths().push_back(path);
    return path;
}

void test_header_and_kv() {
    const std::string blob = build_gguf(
        {{"general.alignment", kU32, 32},
         {"sortformer.encoder.d_model", kU32, 16},
         {"sortformer.encoder.xscaling", kF32, 0, 2.0F},
         {"general.name", kStr, 0, 0.0F, "tiny"},
         {"sortformer.encoder.use_bias", kBool, 1},
         {"some.array", kArr}},
        {},
        32);
    const diar::GgufFile g = diar::GgufFile::load(write_temp(blob));
    expect(g.kv_u32("sortformer.encoder.d_model") == 16, "kv u32");
    expect_near(g.kv_f32("sortformer.encoder.xscaling"), 2.0, 0.0, "kv f32");
    expect(g.kv_string("general.name") == "tiny", "kv string");
    expect(g.has_kv("sortformer.encoder.use_bias"), "kv bool present");
    expect(!g.has_kv("absent.key"), "absent kv");
    expect(g.tensor_count() == 0, "no tensors");
}

void test_tensor_types_and_alignment() {
    // F32 (ne={2,3}): values 0..5 in ggml ne-major order.
    std::vector<std::uint8_t> f32_bytes;
    for (int i = 0; i < 6; i++) put_f32(f32_bytes, static_cast<float>(i));
    // F16 (ne={4}): exactly representable values.
    std::vector<std::uint8_t> f16_bytes;
    for (float v : {0.5F, 1.0F, -2.0F, 0.25F}) {
        const std::uint16_t h = float_to_half_bits(v);
        f16_bytes.push_back(static_cast<std::uint8_t>(h & 0xFF));
        f16_bytes.push_back(static_cast<std::uint8_t>((h >> 8) & 0xFF));
    }
    // Q8_0 (ne={40}): two blocks, d=0.5 then d=-1.0.
    std::vector<std::uint8_t> q8_bytes;
    const auto put_block = [&](float d, int first) {
        const std::uint16_t h = float_to_half_bits(d);
        q8_bytes.push_back(static_cast<std::uint8_t>(h & 0xFF));
        q8_bytes.push_back(static_cast<std::uint8_t>((h >> 8) & 0xFF));
        for (int i = 0; i < 32; i++) {
            const int q = (i % 2 == 0) ? first + i : -(first + i);
            q8_bytes.push_back(static_cast<std::uint8_t>(q));
        }
    };
    put_block(0.5F, 1);   // w = 0.5 * q
    put_block(-1.0F, 3);  // w = -1.0 * q
    // Alignment 64 forces padding between infos-end and data.
    const std::string blob = build_gguf(
        {{"general.alignment", kU32, 64, 0.0F, ""}},
        {{"t.f32", {2, 3}, 0, f32_bytes},
         {"t.f16", {4}, 1, f16_bytes},
         {"t.q8", {40}, 8, q8_bytes}},
        64);
    const diar::GgufFile g = diar::GgufFile::load(write_temp(blob));
    const diar::GgufTensor& f32t = g.at("t.f32");
    expect(f32t.ne.size() == 2 && f32t.ne[0] == 2 && f32t.ne[1] == 3, "f32 ne order");
    for (int i = 0; i < 6; i++)
        expect_near(f32t.data[static_cast<std::size_t>(i)], static_cast<double>(i), 0.0,
                    "f32 value");
    const diar::GgufTensor& f16t = g.at("t.f16");
    expect_near(f16t.data[0], 0.5, 0.0, "f16 0.5");
    expect_near(f16t.data[1], 1.0, 0.0, "f16 1.0");
    expect_near(f16t.data[2], -2.0, 0.0, "f16 -2.0");
    expect_near(f16t.data[3], 0.25, 0.0, "f16 0.25");
    const diar::GgufTensor& q8t = g.at("t.q8");
    expect(q8t.data.size() == 40, "q8 size");
    for (int i = 0; i < 32; i++) {
        const int q = (i % 2 == 0) ? 1 + i : -(1 + i);
        expect_near(q8t.data[static_cast<std::size_t>(i)], 0.5 * static_cast<double>(q), 0.0,
                    "q8 block0");
    }
    for (int i = 0; i < 8; i++) {
        const int q = (i % 2 == 0) ? 3 + i : -(3 + i);
        expect_near(q8t.data[32 + static_cast<std::size_t>(i)], -1.0 * static_cast<double>(q), 0.0,
                    "q8 block1");
    }
    expect(g.find("nope") == nullptr, "find miss");
}

void test_missing_tensor_lists_inventory() {
    std::vector<std::uint8_t> zeros(8, 0);
    const std::string blob = build_gguf({}, {{"a", {2}, 0, zeros}}, 32);
    const diar::GgufFile g = diar::GgufFile::load(write_temp(blob));
    bool threw = false;
    try {
        g.at("b");
    } catch (const diar::WeightFileError& e) {
        threw = true;
        const std::string msg = e.what();
        expect(msg.find("missing tensor 'b'") != std::string::npos, "error names the tensor");
        expect(msg.find(" a") != std::string::npos, "error lists inventory");
    }
    expect(threw, "at(miss) throws");
}

void test_bad_magic_and_truncation() {
    bool threw = false;
    try {
        diar::GgufFile::load(write_temp("NOTGGUF"));
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "bad magic throws");
    threw = false;
    try {
        diar::GgufFile::load("/tmp/diar_gguf_test_no_such_file.bin");
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "missing file throws");
    // Truncated tensor data.
    std::vector<std::uint8_t> f32_bytes(16, 0);
    std::string blob = build_gguf({}, {{"t", {2, 2}, 0, f32_bytes}}, 32);
    blob.resize(blob.size() - 8);
    threw = false;
    try {
        diar::GgufFile::load(write_temp(blob));
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "truncated data throws");
}

void test_half_to_float_edges() {
    using diar::half_to_float;
    expect_near(half_to_float(0x0000), 0.0, 0.0, "half +0");
    expect_near(half_to_float(0x8000), -0.0, 0.0, "half -0");
    expect(std::isinf(half_to_float(0x7C00)), "half +inf");
    expect(std::isnan(half_to_float(0x7E00)), "half nan");
    expect_near(half_to_float(0x3C00), 1.0, 0.0, "half 1");
    expect_near(half_to_float(0xBC00), -1.0, 0.0, "half -1");
    expect_near(half_to_float(0x0001), std::ldexp(1.0, -24), 0.0, "half subnormal min");
    expect_near(half_to_float(0x03FF), std::ldexp(1023.0, -24), 0.0, "half subnormal max");
    expect_near(half_to_float(0x0400), std::ldexp(1.0, -14), 0.0, "half min normal");
    expect_near(half_to_float(0x7BFF), 65504.0, 0.0, "half max");
}

void test_dfw1_roundtrip() {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x31574644u);  // 'DFW1'
    put_u32(b, 1);
    put_u32(b, 2);
    const auto put_name = [&](const std::string& s) {
        put_u32(b, static_cast<std::uint32_t>(s.size()));
        b.insert(b.end(), s.begin(), s.end());
    };
    // tensor 1: "w" dims {2,3} row-major values 0..5
    put_name("w");
    put_u32(b, 2);
    put_u32(b, 2);
    put_u32(b, 3);
    put_u64(b, 6);
    for (int i = 0; i < 6; i++) put_f32(b, static_cast<float>(i));
    // tensor 2: "b" dims {4}
    put_name("b");
    put_u32(b, 1);
    put_u32(b, 4);
    put_u64(b, 4);
    for (int i = 0; i < 4; i++) put_f32(b, -0.25F * static_cast<float>(i));
    const diar::F32WeightFile f =
        diar::F32WeightFile::load(write_temp(std::string(b.begin(), b.end())));
    const diar::F32Tensor& w = f.at("w");
    expect(w.dims.size() == 2 && w.dims[0] == 2 && w.dims[1] == 3, "dfw1 dims");
    for (int i = 0; i < 6; i++)
        expect_near(w.data[static_cast<std::size_t>(i)], static_cast<double>(i), 0.0, "dfw1 value");
    const diar::F32Tensor& bias = f.at("b");
    expect_near(bias.data[3], -0.75, 0.0, "dfw1 bias");
    expect(f.tensor_count() == 2, "dfw1 count");
    // wrong magic must throw
    bool threw = false;
    try {
        diar::F32WeightFile::load(write_temp("XXXX" + std::string(64, '\0')));
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "dfw1 bad magic throws");
}

void test_dfw1_v2_config() {
    const auto build_v2 = [](const std::vector<std::pair<std::string, std::string>>& cfg) {
        std::vector<std::uint8_t> b;
        put_u32(b, 0x31574644u);  // DFW1
        put_u32(b, 2);
        put_u32(b, static_cast<std::uint32_t>(cfg.size()));
        for (const auto& kv : cfg) {
            put_u32(b, static_cast<std::uint32_t>(kv.first.size()));
            b.insert(b.end(), kv.first.begin(), kv.first.end());
            put_u32(b, static_cast<std::uint32_t>(kv.second.size()));
            b.insert(b.end(), kv.second.begin(), kv.second.end());
        }
        put_u32(b, 0);  // 0 tensors
        return std::string(b.begin(), b.end());
    };
    const diar::F32WeightFile f = diar::F32WeightFile::load(write_temp(build_v2(
        {{"sortformer.encoder.d_model", "512"},
            {"sortformer.encoder.xscaling", "1"},
            {"sortformer.scoring.sil_threshold", "0.2"}})));
    expect(f.has_kv("sortformer.encoder.d_model") && !f.has_kv("absent"), "dfw1 v2 has_kv");
    expect(f.kv_u32("sortformer.encoder.d_model") == 512, "dfw1 v2 kv_u32");
    expect(f.kv_u32("sortformer.encoder.xscaling") == 1, "dfw1 v2 numeric bool text");
    expect_near(f.kv_f32("sortformer.scoring.sil_threshold"), 0.2, 1e-6, "dfw1 v2 kv_f32");
    expect(f.kv_string("sortformer.encoder.xscaling") == "1", "dfw1 v2 kv_string");
    bool threw = false;
    try {
        (void)f.kv_u32("sortformer.scoring.sil_threshold");  // "0.2" is not a u32
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "dfw1 v2 typed parse rejects non-u32 text");
    threw = false;
    try {
        (void)f.kv_u32("missing.key");
    } catch (const diar::WeightFileError&) { threw = true; }
    expect(threw, "dfw1 v2 missing key throws");
    expect(f.tensor_count() == 0, "dfw1 v2 zero tensors ok");

    // duplicate config key rejected at load; v1 keeps has_kv == false
    threw = false;
    try {
        diar::F32WeightFile::load(write_temp(build_v2(
            {{"k", "1"}, {"k", "2"}})));
    } catch (const diar::WeightFileError& e) {
        threw = std::string(e.what()).find("duplicate") != std::string::npos;
    }
    expect(threw, "dfw1 v2 duplicate key rejected");
    std::vector<std::uint8_t> v1;
    put_u32(v1, 0x31574644u); put_u32(v1, 1); put_u32(v1, 0);
    const diar::F32WeightFile f1 =
        diar::F32WeightFile::load(write_temp(std::string(v1.begin(), v1.end())));
    expect(!f1.has_kv("anything"), "dfw1 v1 has no config");
}

}  // namespace

int main() {
    test_header_and_kv();
    test_tensor_types_and_alignment();
    test_missing_tensor_lists_inventory();
    test_bad_magic_and_truncation();
    test_half_to_float_edges();
    test_dfw1_roundtrip();
    test_dfw1_v2_config();
    for (const std::string& p : temp_paths()) std::remove(p.c_str());
    std::cout << "PASS: gguf/dfw1 weight reader tests\n";
    return 0;
}
