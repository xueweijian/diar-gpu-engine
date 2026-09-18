// M3 Stage 1 — per-stage wall-clock profiler (compile-time opt-in).
//
// Design contract (docs/M3-P100-CUDA-PLAN.md §2): profiling must have ZERO
// effect on behavior and (when disabled) zero effect on the build. Enabled
// only via -DDIAR_PROFILE_STAGE; every tap compiles to ((void)0) otherwise.
//
// Stage taxonomy (the decision input for which 3-5 CUDA operators cover
// >=80% of chunk time):
//
//   fe            produce_new_mel_frames (engine.cpp)
//   stem          subsampling_forward (mel -> pre-encode embs)
//   concat        [spkcache|fifo|chunk] assembly memcpys
//   xscale        sqrt(D) elementwise on the concat
//   pe            rel-pos table build/slice (per chunk)
//   conformer     17-layer conformer chain (aggregate; per-layer would
//                 add 17x map traffic for identical layers — aggregate is
//                 the decision-relevant number)
//   proj          encoder_proj linear (D -> X)
//   transformer   18-layer post-LN transformer chain (aggregate)
//   head          hidden + speaker linears + sigmoid
//   trim          phantom-row trim copies (engine.cpp, tail chunks)
//   aosc          AoscScoring::update (state rewrite + emitted)
//   gate          BirthGate::append (relabel copy + timeline)
//
// Usage:
//   DIAR_PROFILE_SCOPE("stem");           // RAII: stage = enclosing scope
//   DIAR_PROFILE_SCOPE_PAIR("conformer"); // explicit begin/end
//
// Report: diar::profile::report_json() -> {"stage": {"ns": int, "calls":
// int, "ms_mean": float}, ...} with a "total" row. Caller (k5_runner
// --profile-out) writes it to disk. Not thread-safe by contract: one
// engine per process (k5_runner / Kaggle jobs are separate processes).
#pragma once

#if defined(DIAR_PROFILE_STAGE)

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace diar {
namespace profile {

struct StageStat {
    std::uint64_t ns = 0;
    std::uint64_t calls = 0;
};

// Process-wide accumulator. Intentionally not a singleton-with-static-order
// headaches: function-local static, destroyed at exit after any use.
std::map<std::string, StageStat>& stats();

inline void add(const char* stage, std::uint64_t ns) {
    StageStat& s = stats()[stage];
    s.ns += ns;
    s.calls += 1;
}

class Scope {
public:
    explicit Scope(const char* stage)
        : stage_(stage), t0_(std::chrono::steady_clock::now()) {}
    ~Scope() {
        const auto t1 = std::chrono::steady_clock::now();
        add(stage_, static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0_)
                            .count()));
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    const char* stage_;
    std::chrono::steady_clock::time_point t0_;
};

// JSON report, newest-stage order irrelevant (std::map = alphabetical).
// Doubles printed with %.6f; integers plain. Caller owns the string.
std::string report_json();

}  // namespace profile
}  // namespace diar

#define DIAR_PROFILE_CONCAT_(a, b) a##b
#define DIAR_PROFILE_CONCAT(a, b) DIAR_PROFILE_CONCAT_(a, b)
#define DIAR_PROFILE_SCOPE(stage) \
    ::diar::profile::Scope DIAR_PROFILE_CONCAT(_diar_prof_scope_, __LINE__)(stage)
// Explicit begin/end pairs for sites where a scope would straddle an early
// return (engine.cpp trim sites).
#define DIAR_PROFILE_BEGIN(stage) \
    const auto DIAR_PROFILE_CONCAT(_diar_prof_t0_, __LINE__) = std::chrono::steady_clock::now()
#define DIAR_PROFILE_END(stage) \
    ::diar::profile::add(stage, static_cast<std::uint64_t>( \
        std::chrono::duration_cast<std::chrono::nanoseconds>( \
            std::chrono::steady_clock::now() - DIAR_PROFILE_CONCAT(_diar_prof_t0_, __LINE__)) \
            .count()))

#else  // !DIAR_PROFILE_STAGE

#define DIAR_PROFILE_SCOPE(stage) ((void)0)
#define DIAR_PROFILE_BEGIN(stage) ((void)0)
#define DIAR_PROFILE_END(stage) ((void)0)

#endif  // DIAR_PROFILE_STAGE
