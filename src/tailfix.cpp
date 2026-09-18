// M2 tailfix implementation.
#include "diar/tailfix.hpp"
#include "diar/sortformer.hpp"

namespace diar {

int tail_valid_frames(int feat_len, int subsampling_factor) {
    // Masked valid rows: same chain as sortformer_subsampled_len but driven
    // by the VALID mel length (pre-pad), per NeMo's L1/L2/L3 from feat_len.
    return sortformer_subsampled_len(feat_len, subsampling_factor);
}

}  // namespace diar
