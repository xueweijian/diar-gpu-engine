// M2 Stage 2 rel-pos table — see include/diar/posenc.hpp for pins.
#include "diar/posenc.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace diar {

void relpos_table_forward(float* pos_emb, int l, int d_model) {
    if (l <= 0 || d_model <= 0) return;
    const std::size_t L = static_cast<std::size_t>(l);
    const std::size_t D = static_cast<std::size_t>(d_model);
    const std::size_t P = 2 * L - 1;
    std::vector<float> div(D / 2);
    const float log10000 = std::log(10000.0F);
    for (std::size_t i = 0; i < D / 2; ++i)
        div[i] = std::exp(static_cast<float>(i) * -(log10000 / static_cast<float>(D)));
    for (std::size_t r = 0; r < P; ++r) {
        const float pos = static_cast<float>(L - 1) - static_cast<float>(r);
        for (std::size_t i = 0; i < D / 2; ++i) {
            pos_emb[r * D + 2 * i] = std::sin(pos * div[i]);
            pos_emb[r * D + 2 * i + 1] = std::cos(pos * div[i]);
        }
    }
}

}  // namespace diar
