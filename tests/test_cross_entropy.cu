// ============================================================================
// Softmax cross-entropy: correctness + gradient harness.
//
//   logits (N,V) + targets (N) -> losses (N), lse (N)
//   dlogits[n][v] = (softmax[n][v] - onehot) * dloss
//
// Checks:
//   1. forward vs CPU oracle in double, including a row with logits near 100
//      to prove the max subtraction works (without it that row is inf)
//   2. known answer: uniform logits give exactly log(V)
//   3. confident-correct gives ~0, confident-wrong gives large but FINITE
//   4. dlogits vs oracle
//   5. each row of dlogits sums to 0 (softmax sums to 1, one-hot sums to 1) -
//      cheap, and it catches a missing -1 on the target index
//   6. finite-difference gradient check on the mean loss over logits
//
// Exit 0 = all pass.
// ============================================================================
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/cross_entropy.cuh"

// ---------------------------------------------------------------- CPU oracles

static void cpu_ce_fwd(const float* logits, const int* targets, float* losses, float* lse, int N,
                       int V) {
    for (int n = 0; n < N; ++n) {
        double m = -1e30;
        for (int v = 0; v < V; ++v) m = std::fmax(m, (double)logits[n * V + v]);
        double s = 0.0;
        for (int v = 0; v < V; ++v) s += std::exp((double)logits[n * V + v] - m);
        double l = m + std::log(s);
        lse[n] = (float)l;
        losses[n] = (float)(l - (double)logits[n * V + targets[n]]);
    }
}

static void cpu_ce_bwd(const float* logits, const float* lse, const int* targets, float* dlogits,
                       float dloss, int N, int V) {
    for (int n = 0; n < N; ++n)
        for (int v = 0; v < V; ++v) {
            double p = std::exp((double)logits[n * V + v] - (double)lse[n]);
            double ind = (v == targets[n]) ? 1.0 : 0.0;
            dlogits[n * V + v] = (float)((p - ind) * (double)dloss);
        }
}

// ---------------------------------------------------------------- comparison

static float worst_err(const std::vector<float>& got, const std::vector<float>& ref) {
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float e = std::fabs(got[i] - ref[i]) / (std::fabs(ref[i]) + 1.0f);
        if (e > worst) worst = e;
    }
    return worst;
}

static bool report(const char* label, int N, int V, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  N=%-5d V=%-6d %-30s err=%.2e  %s\n", N, V, label, err, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------- GPU helpers

struct Int_Buffer {
    int* ptr = nullptr;
    explicit Int_Buffer(int n) { cudaMalloc(&ptr, (size_t)n * sizeof(int)); }
    void upload(const int* host, int n) {
        cudaMemcpy(ptr, host, (size_t)n * sizeof(int), cudaMemcpyHostToDevice);
    }
    void free_it() { cudaFree(ptr); }
};

static void gpu_forward(const std::vector<float>& logits, const std::vector<int>& targets,
                        std::vector<float>& losses, std::vector<float>& lse, int N, int V) {
    Device_Buffer d_log(N * V), d_loss(N), d_lse(N);
    Int_Buffer d_tgt(N);
    d_log.upload(const_cast<float*>(logits.data()));
    d_tgt.upload(targets.data(), N);
    launch_crossentropy_fwd(d_log.ptr, d_tgt.ptr, d_loss.ptr, d_lse.ptr, N, V);
    d_loss.download(losses.data());
    d_lse.download(lse.data());
    d_log.free_it();
    d_loss.free_it();
    d_lse.free_it();
    d_tgt.free_it();
}

static void gpu_backward(const std::vector<float>& logits, const std::vector<float>& lse,
                         const std::vector<int>& targets, std::vector<float>& dlogits, float dloss,
                         int N, int V) {
    Device_Buffer d_log(N * V), d_lse(N), d_dlog(N * V);
    Int_Buffer d_tgt(N);
    d_log.upload(const_cast<float*>(logits.data()));
    d_lse.upload(const_cast<float*>(lse.data()));
    d_tgt.upload(targets.data(), N);
    launch_crossentropy_bwd(d_log.ptr, d_lse.ptr, d_tgt.ptr, d_dlog.ptr, dloss, N, V);
    d_dlog.download(dlogits.data());
    d_log.free_it();
    d_lse.free_it();
    d_dlog.free_it();
    d_tgt.free_it();
}

// ---------------------------------------------------------------- the tests

static bool test_forward(int N, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::uniform_int_distribution<int> pick(0, V - 1);
    std::vector<float> logits((size_t)N * V);
    std::vector<int> targets(N);
    for (float& v : logits) v = dist(rng);
    for (int& t : targets) t = pick(rng);
    // one row with huge logits: this is the max-subtraction trap
    for (int v = 0; v < V; ++v) logits[v] += 100.0f;

    std::vector<float> losses(N), lse(N), rlosses(N), rlse(N);
    gpu_forward(logits, targets, losses, lse, N, V);
    cpu_ce_fwd(logits.data(), targets.data(), rlosses.data(), rlse.data(), N, V);

    bool finite = true;
    for (int n = 0; n < N; ++n)
        if (!std::isfinite(losses[n]) || !std::isfinite(lse[n])) finite = false;
    std::printf("  N=%-5d V=%-6d %-30s %s\n", N, V, "all losses finite",
                finite ? "PASS" : "FAIL");

    bool ok = finite;
    ok &= report("fwd  losses", N, V, worst_err(losses, rlosses), 1e-5f);
    ok &= report("fwd  lse", N, V, worst_err(lse, rlse), 1e-5f);
    return ok;
}

static bool test_known_answers(int V) {
    const int N = 3;
    std::vector<float> logits((size_t)N * V, 0.0f);
    std::vector<int> targets(N, 0);
    // row 0: uniform -> loss must be exactly log(V)
    // row 1: confidently correct
    for (int v = 0; v < V; ++v) logits[(size_t)1 * V + v] = (v == 0) ? 30.0f : -30.0f;
    // row 2: confidently wrong
    for (int v = 0; v < V; ++v) logits[(size_t)2 * V + v] = (v == 0) ? -30.0f : 30.0f;

    std::vector<float> losses(N), lse(N);
    gpu_forward(logits, targets, losses, lse, N, V);

    float e_uniform = std::fabs(losses[0] - std::log((float)V));
    bool ok = report("uniform logits -> log(V)", N, V, e_uniform, 1e-4f);
    std::printf("  N=%-5d V=%-6d %-30s got %.4f  %s\n", N, V, "confident correct -> ~0",
                losses[1], losses[1] < 1e-3f ? "PASS" : "FAIL");
    ok &= losses[1] < 1e-3f;
    std::printf("  N=%-5d V=%-6d %-30s got %.4f  %s\n", N, V, "confident wrong -> large, finite",
                losses[2], (losses[2] > 10.0f && std::isfinite(losses[2])) ? "PASS" : "FAIL");
    ok &= losses[2] > 10.0f && std::isfinite(losses[2]);
    return ok;
}

static bool test_backward(int N, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::uniform_int_distribution<int> pick(0, V - 1);
    std::vector<float> logits((size_t)N * V);
    std::vector<int> targets(N);
    for (float& v : logits) v = dist(rng);
    for (int& t : targets) t = pick(rng);

    std::vector<float> losses(N), lse(N);
    gpu_forward(logits, targets, losses, lse, N, V);

    const float dloss = 1.0f / (float)N;
    std::vector<float> g((size_t)N * V), r((size_t)N * V);
    gpu_backward(logits, lse, targets, g, dloss, N, V);
    cpu_ce_bwd(logits.data(), lse.data(), targets.data(), r.data(), dloss, N, V);

    bool ok = report("bwd  dlogits", N, V, worst_err(g, r), 1e-5f);

    float worst_rowsum = 0.0f;
    for (int n = 0; n < N; ++n) {
        double s = 0.0;
        for (int v = 0; v < V; ++v) s += (double)g[(size_t)n * V + v];
        worst_rowsum = std::fmax(worst_rowsum, (float)std::fabs(s));
    }
    ok &= report("dlogits rows sum to 0", N, V, worst_rowsum, 1e-5f);
    return ok;
}

static bool test_finite_difference(int N, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> pick(0, V - 1);
    std::vector<float> logits((size_t)N * V);
    std::vector<int> targets(N);
    for (float& v : logits) v = dist(rng);
    for (int& t : targets) t = pick(rng);

    std::vector<float> losses(N), lse(N);
    gpu_forward(logits, targets, losses, lse, N, V);

    const float dloss = 1.0f / (float)N;
    std::vector<float> g((size_t)N * V);
    gpu_backward(logits, lse, targets, g, dloss, N, V);

    auto mean_loss = [&]() {
        gpu_forward(logits, targets, losses, lse, N, V);
        double s = 0.0;
        for (int n = 0; n < N; ++n) s += (double)losses[n];
        return s / (double)N;
    };

    const float h = 1e-2f;
    float worst = 0.0f;
    for (size_t j = 0; j < logits.size(); ++j) {
        float saved = logits[j];
        logits[j] = saved + h;
        double lp = mean_loss();
        logits[j] = saved - h;
        double lm = mean_loss();
        logits[j] = saved;
        float fd = (float)((lp - lm) / (2.0 * (double)h));
        float e = std::fabs(g[j] - fd) / (std::fabs(fd) + 1.0f);
        if (e > worst) worst = e;
    }
    return report("grad check  dL/dlogits", N, V, worst, 2e-3f);
}

int main() {
    std::mt19937 rng(1234567);
    int failures = 0;

    struct S {
        int N, V;
    };
    S shapes[] = {
        {4, 5},      // tiny
        {7, 11},     // ragged
        {256, 1024}, // realistic block of tokens
        {64, 50257}, // GPT-2 vocab
    };

    std::printf("[cross-entropy forward]\n");
    for (S s : shapes)
        if (!test_forward(s.N, s.V, rng)) ++failures;

    std::printf("[known answers]\n");
    for (int V : {5, 1024})
        if (!test_known_answers(V)) ++failures;

    std::printf("[cross-entropy backward vs CPU oracle]\n");
    for (S s : shapes)
        if (!test_backward(s.N, s.V, rng)) ++failures;

    std::printf("[gradient check vs finite difference, mean loss]\n");
    S small[] = {{4, 5}, {7, 11}};
    for (S s : small)
        if (!test_finite_difference(s.N, s.V, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
