// ============================================================================
// Composed GPT: forward, backward and the optimiser working together.
//
// There is no CPU oracle here. A reference GPT-2 forward and backward in C++
// would be a larger program than the thing it checks. Instead this leans on a
// property that no wrongly-composed model has:
//
//     a correct model can memorise one fixed batch
//
// Freeze one batch of random token ids, run forward/backward/update on that
// same batch a few hundred times, and the loss has to collapse to ~0. The model
// has more parameters than the batch has tokens, so a model whose gradients are
// right will drive the loss into the floor. A model with one wrong cached
// tensor, one missing residual join, or a stale gradient buffer cannot: the
// update direction is wrong, so the loss stalls, wanders, or diverges.
//
// This is what catches the bugs per-op gradient checks structurally cannot see.
// Every op in src/nn is individually correct and finite-difference checked. The
// failure modes left are all about which tensor got handed to which op.
//
// Checks:
//   1. initial loss at real vocab == ln(50257) == 10.82. Proves embedding, the
//      blocks, the tied head and the loss are all wired together, before any
//      gradient exists to be wrong.
//   2. the residual stream actually grows with depth (catches a block whose
//      output is silently discarded, which check 1 does NOT catch)
//   3. loss decreases monotonically over the first 20 steps
//   4. loss < 0.1 by step 300, i.e. the batch is memorised
//   5. no NaN or inf in any parameter at any point
//   6. zero_grad() zeroes: two backwards without a zero between them give a
//      gradient norm exactly twice one backward's
//
// Exit 0 = all pass.
// ============================================================================
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "model/gpt.cuh"

// Token ids are ints; Device_Buffer is floats only.
struct Int_Buffer {
    int* ptr = nullptr;
    int n = 0;
    explicit Int_Buffer(int count) {
        n = count;
        cudaMalloc(&ptr, (size_t)n * sizeof(int));
    }
    void upload(const int* cpu) {
        cudaMemcpy(ptr, cpu, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    }
    void free_it() {
        cudaFree(ptr);
    }
};

static int failures = 0;

static void report(const char* name, bool ok, const char* fmt = nullptr, double a = 0.0, double b = 0.0) {
    char detail[128] = "";
    if (fmt) std::snprintf(detail, sizeof(detail), fmt, a, b);
    std::printf("  %-54s %-26s %s\n", name, detail, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

// Mean and standard deviation of a device buffer, for the residual-growth probe.
static void dev_stats(const float* d, int n, double* mean, double* sd) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += h[i];
    *mean = s / n;
    double q = 0.0;
    for (int i = 0; i < n; ++i) {
        double dv = h[i] - *mean;
        q += dv * dv;
    }
    *sd = std::sqrt(q / n);
}

static bool all_finite(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(h[i])) return false;
    return true;
}

static double grad_norm(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    double q = 0.0;
    for (size_t i = 0; i < n; ++i) q += (double)h[i] * (double)h[i];
    return std::sqrt(q);
}

// ------------------------------------------------------- 1 & 2: initial state

static void test_initial_loss() {
    std::printf("[initial forward, real vocab]\n");

    GPT_Config cfg;
    cfg.max_seq_len = 64;
    cfg.vocab_size = 50257;
    cfg.n_layer = 2;
    cfg.n_head = 4;
    cfg.n_embd = 128;
    const int B = 2, T = 64;

    GPT model;
    model.build(cfg, B, T);
    model.init_random(20260929ull);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> pick(0, cfg.vocab_size - 1);
    std::vector<int> ids(B * T), tgt(B * T);
    for (int& q : ids) q = pick(rng);
    for (int& q : tgt) q = pick(rng);

    Int_Buffer d_ids(B * T), d_tgt(B * T);
    d_ids.upload(ids.data());
    d_tgt.upload(tgt.data());

    float loss = model.forward(d_ids.ptr, d_tgt.ptr);
    double expect = std::log((double)cfg.vocab_size);
    report("initial loss == ln(V)", std::fabs(loss - expect) < 0.05, "got %.4f, want %.4f", loss, expect);

    // A block whose output never reaches the stream leaves the loss at ln(V)
    // too, so check the stream actually did something. After n_layer blocks the
    // residual stream should be visibly wider than the embedding it started as.
    double m0, s0, m1, s1;
    dev_stats(model.acts.encoded, B * T * cfg.n_embd, &m0, &s0);
    dev_stats(model.acts.res3 + (size_t)(cfg.n_layer - 1) * B * T * cfg.n_embd, B * T * cfg.n_embd, &m1, &s1);
    report("residual stream grows with depth", s1 > s0 * 1.05, "sd %.4f -> %.4f", s0, s1);

    model.free_it();
    d_ids.free_it();
    d_tgt.free_it();
}

// ------------------------------------------------------------ 3, 4, 5: overfit

static void test_overfit() {
    std::printf("[memorise one fixed batch]\n");

    // Small vocab keeps 300 steps fast. The composition is what is under test,
    // and it does not care how wide the vocabulary is.
    GPT_Config cfg;
    cfg.max_seq_len = 32;
    cfg.vocab_size = 512;
    cfg.n_layer = 2;
    cfg.n_head = 4;
    cfg.n_embd = 128;
    const int B = 4, T = 32;
    const int steps = 300;

    GPT model;
    model.build(cfg, B, T);
    model.init_random(777ull);

    std::mt19937 rng(99);
    std::uniform_int_distribution<int> pick(0, cfg.vocab_size - 1);
    std::vector<int> ids(B * T), tgt(B * T);
    for (int& q : ids) q = pick(rng);
    for (int& q : tgt) q = pick(rng);

    Int_Buffer d_ids(B * T), d_tgt(B * T);
    d_ids.upload(ids.data());
    d_tgt.upload(tgt.data());

    // No weight decay: it pulls against memorisation, which is the whole point here.
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.0f;

    double first = 0.0, prev = 1e30;
    bool monotonic_early = true, finite = true;
    double loss = 0.0;

    for (int t = 1; t <= steps; ++t) {
        loss = model.forward(d_ids.ptr, d_tgt.ptr);
        if (t == 1) first = loss;
        if (t <= 20 && t > 1 && loss > prev + 1e-6) monotonic_early = false;
        prev = loss;

        model.zero_grad();
        model.backward(d_ids.ptr, d_tgt.ptr);
        model.update(lr, b1, b2, eps, wd, t);

        if (t % 50 == 0 || t == 1) {
            if (!all_finite(model.params, model.n_params)) finite = false;
            std::printf("      step %3d  loss %8.5f\n", t, loss);
        }
    }

    double expect0 = std::log((double)cfg.vocab_size);
    report("step 1 loss == ln(V)", std::fabs(first - expect0) < 0.10, "got %.4f, want %.4f", first, expect0);
    report("loss falls monotonically over first 20 steps", monotonic_early, "final %.5f", loss, 0.0);
    report("batch memorised: loss < 0.1 by step 300", loss < 0.1, "got %.5f", loss, 0.0);
    report("parameters stayed finite", finite);

    model.free_it();
    d_ids.free_it();
    d_tgt.free_it();
}

// ------------------------------------------------------------- 6: zero_grad

static void test_zero_grad() {
    std::printf("[gradient accumulation semantics]\n");

    GPT_Config cfg;
    cfg.max_seq_len = 16;
    cfg.vocab_size = 256;
    cfg.n_layer = 1;
    cfg.n_head = 2;
    cfg.n_embd = 64;
    const int B = 2, T = 16;

    GPT model;
    model.build(cfg, B, T);
    model.init_random(4242ull);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> pick(0, cfg.vocab_size - 1);
    std::vector<int> ids(B * T), tgt(B * T);
    for (int& q : ids) q = pick(rng);
    for (int& q : tgt) q = pick(rng);

    Int_Buffer d_ids(B * T), d_tgt(B * T);
    d_ids.upload(ids.data());
    d_tgt.upload(tgt.data());

    model.forward(d_ids.ptr, d_tgt.ptr);

    model.zero_grad();
    model.backward(d_ids.ptr, d_tgt.ptr);
    double one = grad_norm(model.grads, model.n_params);

    // Second backward with NO zero in between. Every parameter gradient in this
    // engine accumulates, so the buffer must now hold exactly twice as much.
    model.backward(d_ids.ptr, d_tgt.ptr);
    double two = grad_norm(model.grads, model.n_params);

    report("one backward gives nonzero gradient", one > 1e-8, "norm %.6e", one, 0.0);
    report("second backward doubles it (grads accumulate)", std::fabs(two - 2.0 * one) < 1e-3 * two, "%.6e vs %.6e", two, 2.0 * one);

    model.zero_grad();
    double zeroed = grad_norm(model.grads, model.n_params);
    report("zero_grad() clears the buffer", zeroed == 0.0, "norm %.6e", zeroed, 0.0);

    model.free_it();
    d_ids.free_it();
    d_tgt.free_it();
}

int main() {
    test_initial_loss();
    test_overfit();
    test_zero_grad();
    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
