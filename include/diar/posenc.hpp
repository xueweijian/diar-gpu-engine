#pragma once

// M2 Stage 2 relative positional table (NeMo RelPositionalEncoding, offline).
// Reference FP32 implementation; eval mode (dropouts identity).
//
// Upstream pins (NOT assumptions):
//   NeMo PositionalEncoding.create_pe (multi_head_attention.py @ main):
//     div_term = exp(arange(0,d,2) * -(log(10000)/d)), INF_VAL = 10000.0
//     pe[pos,2i] = sin(pos*div[i]); pe[pos,2i+1] = cos(pos*div[i]).
//   NeMo RelPositionalEncoding.extend_pe: positions = arange(L-1,-L,-1)
//     i.e. rows run +(L-1) down to -(L-1) (left-first), 2L-1 rows.
//   NeMo RelPositionalEncoding.forward (cache_len=0): x = x*xscale
//     (xscale = sqrt(d_model), applied by pos_enc — NOT by this function;
//     the caller applies diar::nn::xscale_forward separately, matching the
//     ggml port build_graph_from_embeddings which scales before pos_enc),
//     pos_emb = pe window (2L-1 rows). With a max_len store the window is
//     pe[:, center-L : center+L-1], center = store_rows//2+1 — identical to
//     extend_pe(L) output directly (verified in numpy prototype), so this
//     function builds the 2L-1 rows directly.
//   C++ ggml port RelPositionalEncoding::get_pe_tensor + build_graph
//     (rel_pos_attention.cpp): same sin/cos interleave; diar path loads PE
//     from GGUF (convert_model.py) — the Kaggle teacher gates compare
//     against NeMo-computed pos_emb, so this reference reproduces the NeMo
//     formula, not the GGUF bytes.
//
// Layout: pos_emb [(2L-1), C] row-major. L <= 0 no-op.

namespace diar {

// pos_emb must hold (2*l - 1) * d_model floats.
void relpos_table_forward(float* pos_emb, int l, int d_model);

}  // namespace diar
