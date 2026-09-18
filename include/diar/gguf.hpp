#pragma once

// Minimal GGUF (v2/v3) reader + DFW1 fp32-container reader for weight
// ingestion. No ggml dependency: we only need the file layout, the KV pairs
// the runtime reads (sortformer.* config), and fp32 dequantized tensors.
//
// Supported ggml tensor types (everything else fails loudly with name+type):
//   0 F32, 1 F16, 8 Q8_0 (blocks of 32: fp16 scale d + 32 x int8; w = d*q).
//
// Layout contract mirrors ggml: Tensor::ne[0] is the fastest dimension.
// A PyTorch Linear weight [out,in] is stored with ne = {in, out} and
// row-major bytes (in contiguous), i.e. data()[out][in] — identical to the
// [out,in] row-major contract of include/diar/nn.hpp Linear.
//
// DFW1 is our own fp32 container (K5-A truth anchor: .nemo -> fp32 dump
// written in-kernel by kaggle/m2_stage3/f32bin_writer.py):
//   u32 magic 'DFW1', u32 version=1, u32 n_tensors,
//   per tensor: u32 name_len, name bytes, u32 ndim, u32 dims[ndim]
//               (row-major, dims[0] = outermost), u64 n_floats, f32 data.
// All integers little-endian.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace diar {

// Raised on any malformed input; message names the offender.
class WeightFileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> ne;  // ggml order: ne[0] = fastest dimension
    std::uint32_t ggml_type = 0;
    std::vector<float> data;        // dequantized fp32, ne-major
};

class GgufFile {
public:
    // Reads and parses the whole file eagerly (dequantizes every tensor).
    static GgufFile load(const std::string& path);

    // KV accessors. Numeric getters accept any integer kind (u8..u64, i8..i64,
    // bool) and fail on float/string; kv_f32 accepts f32/f64 and fails
    // otherwise. Missing keys throw with the key name.
    bool has_kv(const std::string& key) const;
    std::uint32_t kv_u32(const std::string& key) const;
    float kv_f32(const std::string& key) const;
    std::string kv_string(const std::string& key) const;

    const GgufTensor* find(const std::string& name) const;
    const GgufTensor& at(const std::string& name) const;  // throws on miss
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    std::unordered_map<std::string, std::size_t> index_;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, std::uint32_t> kv_u32_;
    std::unordered_map<std::string, float> kv_f32_;
    std::unordered_map<std::string, std::string> kv_str_;
};

struct F32Tensor {
    std::string name;
    std::vector<std::uint32_t> dims;  // row-major: dims[0] = outermost
    std::vector<float> data;
};

class F32WeightFile {
public:
    static F32WeightFile load(const std::string& path);

    const F32Tensor* find(const std::string& name) const;
    const F32Tensor& at(const std::string& name) const;  // throws on miss
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    std::unordered_map<std::string, std::size_t> index_;
    std::vector<F32Tensor> tensors_;
};

// IEEE-754 half -> float (handles zero/subnormal/inf/nan). Exposed for tests.
float half_to_float(std::uint16_t h);

}  // namespace diar
