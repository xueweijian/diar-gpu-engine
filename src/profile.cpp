// M3 Stage 1 — profile.hpp implementation (only linked when
// DIAR_PROFILE_STAGE is defined; CMake adds this source to the profile
// build and to k5_runner --profile-out builds).
#include "diar/profile.hpp"

#if defined(DIAR_PROFILE_STAGE)

namespace diar {
namespace profile {

std::map<std::string, StageStat>& stats() {
    static std::map<std::string, StageStat> s;
    return s;
}

std::string report_json() {
    std::string out = "{";
    std::uint64_t total_ns = 0;
    bool first = true;
    for (const auto& kv : stats()) {
        total_ns += kv.second.ns;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "%s\"%s\": {\"ns\": %llu, \"calls\": %llu, \"ms_mean\": %.6f}",
                      first ? "" : ", ", kv.first.c_str(),
                      static_cast<unsigned long long>(kv.second.ns),
                      static_cast<unsigned long long>(kv.second.calls),
                      kv.second.calls ? (static_cast<double>(kv.second.ns) / 1e6)
                                              / static_cast<double>(kv.second.calls)
                                      : 0.0);
        first = false;
        out += buf;
    }
    char tot[96];
    std::snprintf(tot, sizeof(tot),
                  "%s\"total\": {\"ns\": %llu, \"calls\": 0, \"ms_mean\": 0.0}",
                  first ? "" : ", ", static_cast<unsigned long long>(total_ns));
    out += tot;
    out += "}";
    return out;
}

}  // namespace profile
}  // namespace diar

#endif  // DIAR_PROFILE_STAGE
