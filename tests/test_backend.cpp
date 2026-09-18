// M3 Step 2 — CPU backend must be a 1:1 passthrough of the nn:: reference
// (bit identity), and a cuda request must fail loudly without DIAR_WITH_CUDA.
#include "diar/backend.hpp"

#include "diar/nn.hpp"

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using diar::backend::Kind;
using diar::backend::Status;

namespace {

int g_failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

struct Rng {
    std::mt19937 g{1234};
    float operator()() { return std::uniform_real_distribution<float>(-2.f, 2.f)(g); }
};

template <typename T>
void fill(std::vector<float>& v, T& rng) {
    for (auto& e : v) e = rng();
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(),
        a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    auto ctx = diar::backend::create(Kind::cpu);
    expect(ctx->kind() == Kind::cpu, "cpu kind");
    Rng rng;

    {  // linear [260, 512] x [512, 2048] — conformer FF shape family
        diar::backend::LinearOp o{260, 512, 2048};
        std::vector<float> x(o.rows * o.in_dim), w(o.out_dim * o.in_dim),
            b(o.out_dim), y(o.rows * o.out_dim), ref(y.size());
        fill(x, rng); fill(w, rng); fill(b, rng);
        expect(ctx->linear(o, x.data(), w.data(), b.data(), y.data())
              == Status::ok, "linear status");
        diar::nn::linear_forward(x.data(), w.data(), b.data(), ref.data(),
            o.rows, o.in_dim, o.out_dim);
        expect(same(y, ref), "linear bit-identical");
    }
    {  // layernorm (eps pin inside nn::)
        diar::backend::LayerNormOp o{260, 512};
        std::vector<float> x(o.rows * o.dim), g(o.dim), b(o.dim),
            y(o.rows * o.dim), ref(y.size());
        fill(x, rng); fill(g, rng); fill(b, rng);
        expect(ctx->layernorm(o, x.data(), g.data(), b.data(), y.data())
              == Status::ok, "layernorm status");
        diar::nn::layernorm_forward(x.data(), g.data(), b.data(), ref.data(),
            o.rows, o.dim);
        expect(same(y, ref), "layernorm bit-identical");
    }
    {  // softmax (MHA shape family: rows=heads*T, cols=T)
        diar::backend::SoftmaxOp o{16 * 41, 41};
        std::vector<float> x(o.rows * o.cols), y(x.size()), ref(x.size());
        fill(x, rng);
        expect(ctx->softmax(o, x.data(), y.data()) == Status::ok, "softmax status");
        diar::nn::softmax_last_dim(x.data(), ref.data(), o.rows, o.cols);
        expect(same(y, ref), "softmax bit-identical");
    }
    {  // pointwise family
        diar::backend::PointwiseOp o{260 * 512, 0.f};
        std::vector<float> x(o.n), y(o.n), ref(o.n);
        fill(x, rng);
        ctx->pointwise_silu(o, x.data(), y.data());
        diar::nn::silu_forward(x.data(), ref.data(), o.n);
        expect(same(y, ref), "silu bit-identical");
        ctx->pointwise_sigmoid(o, x.data(), y.data());
        diar::nn::sigmoid_forward(x.data(), ref.data(), o.n);
        expect(same(y, ref), "sigmoid bit-identical");
        o.scale = 22.627417f;  // sqrt(512)
        ctx->pointwise_xscale(o, x.data(), y.data());
        diar::nn::xscale_forward(x.data(), ref.data(), o.n, o.scale);
        expect(same(y, ref), "xscale bit-identical");
        std::vector<float> xin(o.n), rin(o.n);
        fill(xin, rng); rin = xin;
        ctx->pointwise_relu(o, xin.data());
        diar::nn::relu_inplace(rin.data(), o.n);
        expect(same(xin, rin), "relu bit-identical");
    }
    {  // glu
        diar::backend::GluOp o{41, 512};
        std::vector<float> x(o.t * o.c * 2), y(o.t * o.c), ref(y.size());
        fill(x, rng);
        expect(ctx->glu(o, x.data(), y.data()) == Status::ok, "glu status");
        diar::nn::glu_forward(x.data(), ref.data(), o.t, o.c);
        expect(same(y, ref), "glu bit-identical");
    }
    {  // conv1d + depthwise (conformer conv k9)
        diar::backend::Conv1dOp o{41, 512, 1024, 9, 1, 4, false};
        std::vector<float> x(o.t_in * o.c_in), w(o.c_out * o.c_in * o.k),
            b(o.c_out), y(o.t_in * o.c_out), ref(y.size());
        fill(x, rng); fill(w, rng); fill(b, rng);
        expect(ctx->conv1d(o, x.data(), w.data(), b.data(), y.data())
              == Status::ok, "conv1d status");
        diar::nn::conv1d_forward(x.data(), w.data(), b.data(), ref.data(),
            o.t_in, o.c_in, o.c_out, o.k, o.stride, o.pad);
        expect(same(y, ref), "conv1d bit-identical");

        diar::backend::Conv1dOp dw{41, 512, 512, 9, 1, 4, true};
        std::vector<float> xw(dw.t_in * dw.c_in), ww(dw.c_in * dw.k),
            bw(dw.c_in), yw(dw.t_in * dw.c_in), rw(yw.size());
        fill(xw, rng); fill(ww, rng); fill(bw, rng);
        expect(ctx->conv1d(dw, xw.data(), ww.data(), bw.data(), yw.data())
              == Status::ok, "dw status");
        diar::nn::conv1d_depthwise_forward(xw.data(), ww.data(), bw.data(),
            rw.data(), dw.t_in, dw.c_in, dw.k, dw.stride, dw.pad);
        expect(same(yw, rw), "dw bit-identical");
    }
    {  // conv2d (stem k3s2p1 shape family)
        diar::backend::Conv2dOp o{64, 64, 10, 160, 3, 2, 1};
        std::vector<float> x(o.c_in * o.h_in * o.w_in),
            w(o.c_out * o.c_in * o.k * o.k), b(o.c_out),
            y(o.c_out * 10 * 80), ref(y.size());
        fill(x, rng); fill(w, rng); fill(b, rng);
        expect(ctx->conv2d(o, x.data(), w.data(), b.data(), y.data())
              == Status::ok, "conv2d status");
        diar::nn::conv2d_forward(x.data(), w.data(), b.data(), ref.data(),
            o.c_in, o.h_in, o.w_in, o.c_out, o.k, o.k, o.stride, o.stride,
            o.pad, o.pad);
        expect(same(y, ref), "conv2d bit-identical");
    }
    {  // cuda request fails loudly in an OFF build
        bool threw = false;
        try { (void)diar::backend::create(Kind::cuda); }
        catch (const std::runtime_error&) { threw = true; }
        expect(threw, "cuda create throws without DIAR_WITH_CUDA");
    }
    expect(std::strcmp(diar::backend::status_name(Status::ok), "ok") == 0,
          "status_name ok");
    if (g_failures == 0) std::cout << "test_backend ok\n";
    return g_failures == 0 ? 0 : 1;
}
