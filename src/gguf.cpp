#include "diar/gguf.hpp"

#include <cstring>
#include <fstream>

namespace diar {
namespace {

[[noreturn]] void bad(const std::string& what) { throw WeightFileError(what); }

std::vector<std::uint8_t> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) bad("cannot open weight file: " + path);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>());
}

struct Cursor {
    const std::uint8_t* p;
    const std::uint8_t* end;

    void need(std::size_t n, const char* what) const {
        if (static_cast<std::size_t>(end - p) < n)
            bad(std::string("truncated file while reading ") + what);
    }
    std::uint8_t u8(const char* w) { need(1, w); return *p++; }
    std::uint16_t u16(const char* w) { need(2, w); std::uint16_t v; std::memcpy(&v, p, 2); p += 2; return v; }
    std::uint32_t u32(const char* w) { need(4, w); std::uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    std::uint64_t u64(const char* w) { need(8, w); std::uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    float f32(const char* w) { std::uint32_t v = u32(w); float f; std::memcpy(&f, &v, 4); return f; }
    double f64(const char* w) { std::uint64_t v = u64(w); double f; std::memcpy(&f, &v, 8); return f; }
    std::string str(const char* w) {
        const std::uint64_t n = u64(w);
        need(n, w);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
};

enum GgufValueType : std::uint32_t {
    kU8 = 0, kI8, kU16, kI16, kU32, kI32, kF32, kBool, kString, kArray, kU64, kI64, kF64,
};

std::uint32_t read_scalar_u32(Cursor& c, std::uint32_t type, const std::string& key) {
    switch (type) {
        case kU8: return c.u8("kv u8");
        case kI8: return static_cast<std::uint32_t>(static_cast<int8_t>(c.u8("kv i8")));
        case kU16: return c.u16("kv u16");
        case kI16: return static_cast<std::uint32_t>(static_cast<int16_t>(c.u16("kv i16")));
        case kU32: return c.u32("kv u32");
        case kI32: return c.u32("kv i32");
        case kBool: return c.u8("kv bool") ? 1u : 0u;
        case kU64: {
            const std::uint64_t v = c.u64("kv u64");
            if (v > 0xFFFFFFFFull) bad("kv u64 overflows u32: " + key);
            return static_cast<std::uint32_t>(v);
        }
        case kI64: {
            const std::int64_t v = static_cast<std::int64_t>(c.u64("kv i64"));
            if (v < 0 || v > 0xFFFFFFFFll) bad("kv i64 overflows u32: " + key);
            return static_cast<std::uint32_t>(v);
        }
        default: bad("kv '" + key + "' is not integer-kind (type " + std::to_string(type) + ")");
    }
}

void skip_array(Cursor& c, std::uint32_t elem_type, std::uint64_t count) {
    switch (elem_type) {
        case kU8: case kI8: case kBool: c.need(count, "array bytes"); c.p += count; return;
        case kU16: case kI16: c.need(count * 2, "array bytes"); c.p += count * 2; return;
        case kU32: case kI32: case kF32: c.need(count * 4, "array bytes"); c.p += count * 4; return;
        case kU64: case kI64: case kF64: c.need(count * 8, "array bytes"); c.p += count * 8; return;
        case kString:
            for (std::uint64_t i = 0; i < count; i++) c.str("array string");
            return;
        case kArray: {
            for (std::uint64_t i = 0; i < count; i++) {
                const std::uint32_t t = c.u32("nested array type");
                const std::uint64_t n = c.u64("nested array count");
                skip_array(c, t, n);
            }
            return;
        }
        default: bad("unsupported array element type " + std::to_string(elem_type));
    }
}

std::size_t ggml_type_size(std::uint32_t type) {
    switch (type) {
        case 0: return 4;   // F32
        case 1: return 2;   // F16
        case 8: return 34;  // Q8_0: fp16 d + 32 x int8
        default: bad("unsupported ggml tensor type " + std::to_string(type) +
                     " (supported: 0 F32, 1 F16, 8 Q8_0)");
    }
}

std::uint64_t ggml_nbytes(const GgufTensor& t) {
    std::uint64_t n = 1;
    for (const std::uint64_t d : t.ne) n *= d;
    switch (t.ggml_type) {
        case 0: return n * 4;
        case 1: return n * 2;
        case 8: return ((n + 31) / 32) * 34;
        default: bad("unsupported ggml tensor type " + std::to_string(t.ggml_type));
    }
}

void dequantize(GgufTensor& t, const std::uint8_t* src, std::size_t len) {
    const std::uint64_t n = ggml_nbytes(t);
    if (len < n) bad("tensor '" + t.name + "' truncated on disk");
    std::uint64_t count = 1;
    for (const std::uint64_t d : t.ne) count *= d;
    t.data.resize(static_cast<std::size_t>(count));
    switch (t.ggml_type) {
        case 0:
            if (len < count * 4) bad("tensor '" + t.name + "' truncated");
            std::memcpy(t.data.data(), src, static_cast<std::size_t>(count) * 4);
            return;
        case 1:
            for (std::uint64_t i = 0; i < count; i++) {
                std::uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                t.data[static_cast<std::size_t>(i)] = half_to_float(h);
            }
            return;
        case 8: {
            std::size_t out = 0;
            for (std::uint64_t blk = 0; blk < count; blk += 32) {
                std::uint16_t h;
                std::memcpy(&h, src, 2);
                src += 2;
                const float d = half_to_float(h);
                for (int i = 0; i < 32 && out < count; i++, out++, src++)
                    t.data[out] = d * static_cast<float>(static_cast<std::int8_t>(*src));
            }
            return;
        }
        default: bad("unsupported ggml tensor type " + std::to_string(t.ggml_type));
    }
}

}  // namespace

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t man = h & 0x03FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;  // +-zero
        } else {
            int k = 9;
            while (((man >> k) & 1u) == 0) k--;
            const std::uint32_t frac = (man << (10 - k)) & 0x3FFu;
            bits = sign | (static_cast<std::uint32_t>(k + 103) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);  // inf / nan
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

GgufFile GgufFile::load(const std::string& path) {
    const std::vector<std::uint8_t> bytes = slurp(path);
    Cursor c{bytes.data(), bytes.data() + bytes.size()};

    if (c.u32("magic") != 0x46554747u) bad("not a GGUF file (bad magic): " + path);
    const std::uint32_t version = c.u32("version");
    if (version != 2 && version != 3) bad("unsupported GGUF version " + std::to_string(version));
    const std::uint64_t tensor_count = c.u64("tensor count");
    const std::uint64_t kv_count = c.u64("kv count");

    GgufFile f;
    std::uint32_t alignment = 32;
    for (std::uint64_t i = 0; i < kv_count; i++) {
        const std::string key = c.str("kv key");
        const std::uint32_t type = c.u32("kv type");
        if (type == kString) {
            f.kv_str_[key] = c.str("kv string");
        } else if (type == kF32) {
            f.kv_f32_[key] = c.f32("kv f32");
        } else if (type == kF64) {
            f.kv_f32_[key] = static_cast<float>(c.f64("kv f64"));
        } else if (type == kArray) {
            const std::uint32_t elem = c.u32("array elem type");
            const std::uint64_t n = c.u64("array count");
            skip_array(c, elem, n);
        } else {
            f.kv_u32_[key] = read_scalar_u32(c, type, key);
        }
    }
    if (auto it = f.kv_u32_.find("general.alignment"); it != f.kv_u32_.end() && it->second != 0)
        alignment = it->second;

    struct Info { std::string name; std::vector<std::uint64_t> ne; std::uint32_t type; std::uint64_t off; };
    std::vector<Info> infos;
    infos.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; i++) {
        Info info;
        info.name = c.str("tensor name");
        const std::uint32_t ndim = c.u32("tensor ndim");
        if (ndim == 0 || ndim > 4) bad("tensor '" + info.name + "' ndim out of range");
        info.ne.resize(ndim);
        for (std::uint32_t d = 0; d < ndim; d++) info.ne[d] = c.u64("tensor dim");
        info.type = c.u32("tensor type");
        ggml_type_size(info.type);  // fail loudly on unsupported types here
        info.off = c.u64("tensor offset");
        infos.push_back(std::move(info));
    }

    std::size_t data_start = static_cast<std::size_t>(c.p - bytes.data());
    data_start = (data_start + alignment - 1) / alignment * alignment;
    if (bytes.size() < data_start) bad("data section beyond end of file");

    f.tensors_.resize(infos.size());
    for (std::size_t i = 0; i < infos.size(); i++) {
        GgufTensor& t = f.tensors_[i];
        t.name = infos[i].name;
        t.ne = infos[i].ne;
        t.ggml_type = infos[i].type;
        if (infos[i].off > bytes.size() - data_start) bad("tensor '" + t.name + "' offset out of range");
        dequantize(t, bytes.data() + data_start + infos[i].off,
                   bytes.size() - data_start - infos[i].off);
        f.index_[t.name] = i;
    }
    return f;
}

bool GgufFile::has_kv(const std::string& key) const {
    return kv_u32_.count(key) || kv_f32_.count(key) || kv_str_.count(key);
}
std::uint32_t GgufFile::kv_u32(const std::string& key) const {
    const auto i = kv_u32_.find(key);
    if (i == kv_u32_.end()) bad("missing integer kv: " + key);
    return i->second;
}
float GgufFile::kv_f32(const std::string& key) const {
    const auto i = kv_f32_.find(key);
    if (i == kv_f32_.end()) bad("missing float kv: " + key);
    return i->second;
}
std::string GgufFile::kv_string(const std::string& key) const {
    const auto i = kv_str_.find(key);
    if (i == kv_str_.end()) bad("missing string kv: " + key);
    return i->second;
}
const GgufTensor* GgufFile::find(const std::string& name) const {
    const auto i = index_.find(name);
    return i == index_.end() ? nullptr : &tensors_[i->second];
}
const GgufTensor& GgufFile::at(const std::string& name) const {
    const GgufTensor* t = find(name);
    if (t == nullptr) {
        std::string have;
        for (const GgufTensor& e : tensors_) have += " " + e.name;
        bad("missing tensor '" + name + "'; have:" + have);
    }
    return *t;
}

F32WeightFile F32WeightFile::load(const std::string& path) {
    const std::vector<std::uint8_t> bytes = slurp(path);
    Cursor c{bytes.data(), bytes.data() + bytes.size()};
    if (c.u32("magic") != 0x31574644u) bad("not a DFW1 file (bad magic): " + path);
    const std::uint32_t version = c.u32("version");
    if (version != 1 && version != 2) bad("unsupported DFW1 version");

    F32WeightFile f;
    if (version == 2) {
        const std::uint32_t n_cfg = c.u32("config pair count");
        for (std::uint32_t i = 0; i < n_cfg; i++) {
            std::string key, val;
            for (int which = 0; which < 2; which++) {
                const std::uint32_t len = c.u32(which ? "value len" : "key len");
                c.need(len, which ? "config value" : "config key");
                std::string& dst = which ? val : key;
                dst.assign(reinterpret_cast<const char*>(c.p), len);
                c.p += len;
            }
            if (!f.cfg_.emplace(key, val).second) bad("duplicate config key '" + key + "'");
        }
    }
    const std::uint32_t n_tensors = c.u32("tensor count");
    f.tensors_.resize(n_tensors);
    for (std::uint32_t i = 0; i < n_tensors; i++) {
        F32Tensor& t = f.tensors_[i];
        const std::uint32_t name_len = c.u32("name len");
        c.need(name_len, "tensor name");
        t.name.assign(reinterpret_cast<const char*>(c.p), name_len);
        c.p += name_len;
        const std::uint32_t ndim = c.u32("ndim");
        if (ndim == 0 || ndim > 4) bad("tensor '" + t.name + "' ndim out of range");
        t.dims.resize(ndim);
        std::uint64_t count = 1;
        for (std::uint32_t d = 0; d < ndim; d++) {
            t.dims[d] = c.u32("dim");
            count *= t.dims[d];
        }
        const std::uint64_t stored = c.u64("n_floats");
        if (stored != count) bad("tensor '" + t.name + "' n_floats mismatch");
        c.need(count * 4, "tensor data");
        t.data.resize(static_cast<std::size_t>(count));
        std::memcpy(t.data.data(), c.p, static_cast<std::size_t>(count) * 4);
        c.p += count * 4;
        f.index_[t.name] = i;
    }
    return f;
}

const F32Tensor* F32WeightFile::find(const std::string& name) const {
    const auto i = index_.find(name);
    return i == index_.end() ? nullptr : &tensors_[i->second];
}
const F32Tensor& F32WeightFile::at(const std::string& name) const {
    const F32Tensor* t = find(name);
    if (t == nullptr) {
        std::string have;
        for (const F32Tensor& e : tensors_) have += " " + e.name;
        bad("missing tensor '" + name + "'; have:" + have);
    }
    return *t;
}

}  // namespace diar

namespace diar {
std::vector<std::string> GgufFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const GgufTensor& t : tensors_) names.push_back(t.name);
    return names;
}

std::vector<std::string> F32WeightFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const F32Tensor& t : tensors_) names.push_back(t.name);
    return names;
}

bool F32WeightFile::has_kv(const std::string& key) const {
    return cfg_.find(key) != cfg_.end();
}

std::uint32_t F32WeightFile::kv_u32(const std::string& key) const {
    const auto it = cfg_.find(key);
    if (it == cfg_.end()) throw WeightFileError("DFW1 config missing key: " + key);
    try {
        std::size_t used = 0;
        const unsigned long v = std::stoul(it->second, &used, 10);
        if (used != it->second.size() || v > 0xFFFFFFFFul)
            throw std::invalid_argument("range");
        return static_cast<std::uint32_t>(v);
    } catch (const std::exception&) {
        throw WeightFileError("DFW1 config key '" + key + "' is not a u32: '" + it->second + "'");
    }
}

float F32WeightFile::kv_f32(const std::string& key) const {
    const auto it = cfg_.find(key);
    if (it == cfg_.end()) throw WeightFileError("DFW1 config missing key: " + key);
    try {
        std::size_t used = 0;
        const float v = std::stof(it->second, &used);
        if (used != it->second.size()) throw std::invalid_argument("trailing");
        return v;
    } catch (const std::exception&) {
        throw WeightFileError("DFW1 config key '" + key + "' is not an f32: '" + it->second + "'");
    }
}

std::string F32WeightFile::kv_string(const std::string& key) const {
    const auto it = cfg_.find(key);
    if (it == cfg_.end()) throw WeightFileError("DFW1 config missing key: " + key);
    return it->second;
}

}  // namespace diar
