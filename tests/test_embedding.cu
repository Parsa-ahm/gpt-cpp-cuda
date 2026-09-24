// ============================================================================
// Embedding (token + position): correctness + gradient harness.
//
//   out[b][t][c] = wte[ids[b][t]][c] + wpe[t][c]
//
// Checks:
//   1. forward vs CPU oracle, with a deliberately collision-heavy id stream
//   2. dwte vs oracle, ACCUMULATING (this is the atomicAdd scatter)
//   3. dwpe vs oracle, ACCUMULATING
//   4. run-to-run consistency: backward twice, results agree to 1e-4 RELATIVE,
//      not bitwise. atomicAdd on floats reorders, float add is not associative,
//      so bitwise reproducibility is not a property this op has. Documenting
//      that is better than pretending otherwise.
//   5. finite-difference gradient check on L = sum(dout * out) over wte and wpe
//
// There is no dx. The input is integer ids; no gradient flows into them.
//
// Exit 0 = all pass.
// ============================================================================
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/embedding.cuh"

// ---------------------------------------------------------------- CPU oracles

static void cpu_emb_fwd(const int* ids, const float* wte, const float* wpe, float* out, int B,
                        int T, int C) {
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            int id = ids[b * T + t];
            for (int c = 0; c < C; ++c)
                out[(b * T + t) * C + c] = wte[id * C + c] + wpe[t * C + c];
        }
}

static void cpu_emb_bwd(const int* ids, const float* dout, float* dwte, float* dwpe, int B, int T,
                        int C, int V) {
    std::vector<double> awte((size_t)V * C, 0.0), awpe((size_t)T * C, 0.0);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            int id = ids[b * T + t];
            for (int c = 0; c < C; ++c) {
                double g = (double)dout[(b * T + t) * C + c];
                awte[(size_t)id * C + c] += g;
                awpe[(size_t)t * C + c] += g;
            }
        }
    for (size_t i = 0; i < awte.size(); ++i) dwte[i] += (float)awte[i];
    for (size_t i = 0; i < awpe.size(); ++i) dwpe[i] += (float)awpe[i];
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

static bool report(const char* label, int B, int T, int C, int V, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  B=%-3d T=%-4d C=%-4d V=%-6d %-24s err=%.2e  %s\n", B, T, C, V, label, err,
                ok ? "PASS" : "FAIL");
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

static void gpu_forward(const std::vector<int>& ids, const std::vector<float>& wte,
                        const std::vector<float>& wpe, std::vector<float>& out, int B, int T, int C,
                        int V) {
    Int_Buffer d_ids(B * T);
    Device_Buffer d_wte(V * C), d_wpe(T * C), d_out(B * T * C);
    d_ids.upload(ids.data(), B * T);
    d_wte.upload(const_cast<float*>(wte.data()));
    d_wpe.upload(const_cast<float*>(wpe.data()));
    launch_embedding_fwd(d_ids.ptr, d_wte.ptr, d_wpe.ptr, d_out.ptr, B, T, C);
    d_out.download(out.data());
    d_ids.free_it();
    d_wte.free_it();
    d_wpe.free_it();
    d_out.free_it();
}

static void gpu_backward(const std::vector<int>& ids, const std::vector<float>& dout,
                         const std::vector<float>& seed_wte, const std::vector<float>& seed_wpe,
                         std::vector<float>& gwte, std::vector<float>& gwpe, int B, int T, int C,
                         int V) {
    Int_Buffer d_ids(B * T);
    Device_Buffer d_dout(B * T * C), d_dwte(V * C), d_dwpe(T * C);
    d_ids.upload(ids.data(), B * T);
    d_dout.upload(const_cast<float*>(dout.data()));
    d_dwte.upload(const_cast<float*>(seed_wte.data()));
    d_dwpe.upload(const_cast<float*>(seed_wpe.data()));
    launch_embedding_bwd(d_ids.ptr, d_dout.ptr, d_dwte.ptr, d_dwpe.ptr, B, T, C);
    d_dwte.download(gwte.data());
    d_dwpe.download(gwpe.data());
    d_ids.free_it();
    d_dout.free_it();
    d_dwte.free_it();
    d_dwpe.free_it();
}

// Small vocab on purpose: forces many tokens to share a row of wte, which is
// exactly the collision case the atomicAdd exists for.
static std::vector<int> make_ids(int B, int T, int V, std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, V - 1);
    std::vector<int> ids(B * T);
    for (int& v : ids) v = pick(rng);
    return ids;
}

// ---------------------------------------------------------------- the tests

static bool test_forward(int B, int T, int C, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<int> ids = make_ids(B, T, V, rng);
    std::vector<float> wte((size_t)V * C), wpe((size_t)T * C), out((size_t)B * T * C),
        ref((size_t)B * T * C);
    for (float& v : wte) v = dist(rng);
    for (float& v : wpe) v = dist(rng);

    gpu_forward(ids, wte, wpe, out, B, T, C, V);
    cpu_emb_fwd(ids.data(), wte.data(), wpe.data(), ref.data(), B, T, C);
    return report("fwd  wte[id] + wpe[t]", B, T, C, V, worst_err(out, ref), 1e-6f);
}

static bool test_backward(int B, int T, int C, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<int> ids = make_ids(B, T, V, rng);
    std::vector<float> dout((size_t)B * T * C);
    for (float& v : dout) v = dist(rng);

    std::vector<float> junk_wte((size_t)V * C), junk_wpe((size_t)T * C);
    for (float& v : junk_wte) v = dist(rng);
    for (float& v : junk_wpe) v = dist(rng);

    std::vector<float> gwte((size_t)V * C), gwpe((size_t)T * C);
    gpu_backward(ids, dout, junk_wte, junk_wpe, gwte, gwpe, B, T, C, V);

    std::vector<float> rwte = junk_wte, rwpe = junk_wpe;
    cpu_emb_bwd(ids.data(), dout.data(), rwte.data(), rwpe.data(), B, T, C, V);

    bool ok = report("bwd  dwte += (scatter)", B, T, C, V, worst_err(gwte, rwte), 1e-4f);
    ok &= report("bwd  dwpe +=", B, T, C, V, worst_err(gwpe, rwpe), 1e-4f);
    return ok;
}

// atomicAdd reorders, so this is a tolerance check, not a bitwise one.
static bool test_run_to_run(int B, int T, int C, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<int> ids = make_ids(B, T, V, rng);
    std::vector<float> dout((size_t)B * T * C);
    for (float& v : dout) v = dist(rng);
    std::vector<float> zwte((size_t)V * C, 0.0f), zwpe((size_t)T * C, 0.0f);

    std::vector<float> a_wte((size_t)V * C), a_wpe((size_t)T * C);
    std::vector<float> b_wte((size_t)V * C), b_wpe((size_t)T * C);
    gpu_backward(ids, dout, zwte, zwpe, a_wte, a_wpe, B, T, C, V);
    gpu_backward(ids, dout, zwte, zwpe, b_wte, b_wpe, B, T, C, V);

    bool ok = report("two runs agree (dwte)", B, T, C, V, worst_err(a_wte, b_wte), 1e-4f);
    ok &= report("two runs agree (dwpe)", B, T, C, V, worst_err(a_wpe, b_wpe), 1e-4f);
    return ok;
}

static bool test_finite_difference(int B, int T, int C, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<int> ids = make_ids(B, T, V, rng);
    std::vector<float> wte((size_t)V * C), wpe((size_t)T * C), dout((size_t)B * T * C);
    for (float& v : wte) v = dist(rng);
    for (float& v : wpe) v = dist(rng);
    for (float& v : dout) v = dist(rng);

    std::vector<float> zwte((size_t)V * C, 0.0f), zwpe((size_t)T * C, 0.0f);
    std::vector<float> gwte((size_t)V * C), gwpe((size_t)T * C);
    gpu_backward(ids, dout, zwte, zwpe, gwte, gwpe, B, T, C, V);

    std::vector<float> o((size_t)B * T * C);
    auto loss = [&]() {
        gpu_forward(ids, wte, wpe, o, B, T, C, V);
        double s = 0.0;
        for (size_t i = 0; i < o.size(); ++i) s += (double)dout[i] * (double)o[i];
        return s;
    };

    const float h = 1e-2f;
    auto sweep = [&](std::vector<float>& p, const std::vector<float>& analytic, const char* name) {
        float worst = 0.0f;
        for (size_t j = 0; j < p.size(); ++j) {
            float saved = p[j];
            p[j] = saved + h;
            double lp = loss();
            p[j] = saved - h;
            double lm = loss();
            p[j] = saved;
            float fd = (float)((lp - lm) / (2.0 * (double)h));
            float e = std::fabs(analytic[j] - fd) / (std::fabs(fd) + 1.0f);
            if (e > worst) worst = e;
        }
        return report(name, B, T, C, V, worst, 2e-3f);
    };

    bool ok = sweep(wte, gwte, "grad check  dL/dwte");
    ok &= sweep(wpe, gwpe, "grad check  dL/dwpe");
    return ok;
}

int main() {
    std::mt19937 rng(90210);
    int failures = 0;

    struct S {
        int B, T, C, V;
    };
    S shapes[] = {
        {2, 4, 3, 5},       // tiny, heavy collisions
        {3, 7, 5, 4},       // more tokens than vocab: every row of wte gets hit
        {4, 64, 384, 512},  // GPT-2 small width
        {2, 128, 768, 1024} // GPT-2 124M width
    };

    std::printf("[embedding forward]\n");
    for (S s : shapes)
        if (!test_forward(s.B, s.T, s.C, s.V, rng)) ++failures;

    std::printf("[embedding backward vs CPU oracle, accumulating into pre-filled grads]\n");
    for (S s : shapes)
        if (!test_backward(s.B, s.T, s.C, s.V, rng)) ++failures;

    std::printf("[atomics: two identical runs agree to tolerance, not bitwise]\n");
    for (S s : shapes)
        if (!test_run_to_run(s.B, s.T, s.C, s.V, rng)) ++failures;

    std::printf("[gradient check vs finite difference, L = sum(dout * out)]\n");
    S small[] = {{2, 4, 3, 5}, {3, 7, 5, 4}};
    for (S s : small)
        if (!test_finite_difference(s.B, s.T, s.C, s.V, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
