// ============================================================================
// AdamW: correctness harness.
//
// Checks:
//   1. one step from zeroed state vs CPU oracle, several lr / weight_decay
//   2. fifty steps vs oracle, so drift cannot accumulate unnoticed
//   3. BIAS CORRECTION: with constant g, step 1 must move p by roughly
//      lr * (1 + wd * p). Without the 1/(1-beta^t) terms it moves by about a
//      tenth of that, and this test says so loudly.
//   4. DECOUPLING: AdamW (wd applied to p) must NOT equal Adam-with-L2 (wd
//      folded into g). If these two agree you built the wrong optimiser.
//   5. convergence: minimise sum(x^2), loss falls monotonically to ~0
//   6. m and v match the oracle at the end, not just p
//
// Exit 0 = all pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "optim/adamw.cuh"

static const float B1 = 0.9f, B2 = 0.95f, EPS = 1e-8f;

// ---------------------------------------------------------------- CPU oracle

static void cpu_adamw(float* p, const float* g, float* m, float* v, int n, float lr, float beta1,
                      float beta2, float eps, float wd, int t) {
    double c1 = 1.0 - std::pow((double)beta1, (double)t);
    double c2 = 1.0 - std::pow((double)beta2, (double)t);
    for (int i = 0; i < n; ++i) {
        m[i] = (float)((double)beta1 * m[i] + (1.0 - beta1) * (double)g[i]);
        v[i] = (float)((double)beta2 * v[i] + (1.0 - beta2) * (double)g[i] * (double)g[i]);
        double mhat = (double)m[i] / c1;
        double vhat = (double)v[i] / c2;
        p[i] = (float)((double)p[i] - (double)lr * (mhat / (std::sqrt(vhat) + (double)eps) +
                                                    (double)wd * (double)p[i]));
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

static bool report(const char* label, int n, float lr, float wd, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  n=%-6d lr=%-7.4f wd=%-5.2f %-24s err=%.2e  %s\n", n, lr, wd, label, err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------- GPU helper

// Runs `steps` AdamW updates on the device, leaving the results in p, m, v.
static void gpu_run(std::vector<float>& p, const std::vector<float>& g, std::vector<float>& m,
                    std::vector<float>& v, float lr, float wd, int steps) {
    int n = (int)p.size();
    Device_Buffer d_p(n), d_g(n), d_m(n), d_v(n);
    d_p.upload(p.data());
    d_g.upload(const_cast<float*>(g.data()));
    d_m.upload(m.data());
    d_v.upload(v.data());
    for (int t = 1; t <= steps; ++t)
        launch_adamw_step(d_p.ptr, d_g.ptr, d_m.ptr, d_v.ptr, n, lr, B1, B2, EPS, wd, t);
    d_p.download(p.data());
    d_m.download(m.data());
    d_v.download(v.data());
    d_p.free_it();
    d_g.free_it();
    d_m.free_it();
    d_v.free_it();
}

// ---------------------------------------------------------------- the tests

static bool test_steps(int n, float lr, float wd, int steps, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> p(n), g(n);
    for (float& x : p) x = dist(rng);
    for (float& x : g) x = dist(rng);

    std::vector<float> gm(n, 0.0f), gv(n, 0.0f), gp = p;
    gpu_run(gp, g, gm, gv, lr, wd, steps);

    std::vector<float> rm(n, 0.0f), rv(n, 0.0f), rp = p;
    for (int t = 1; t <= steps; ++t)
        cpu_adamw(rp.data(), g.data(), rm.data(), rv.data(), n, lr, B1, B2, EPS, wd, t);

    char label[64];
    std::snprintf(label, sizeof(label), "%d step%s: p", steps, steps == 1 ? "" : "s");
    bool ok = report(label, n, lr, wd, worst_err(gp, rp), 1e-5f);
    ok &= report("        m", n, lr, wd, worst_err(gm, rm), 1e-5f);
    ok &= report("        v", n, lr, wd, worst_err(gv, rv), 1e-5f);
    return ok;
}

// Step 1 with bias correction: mhat/sqrt(vhat) is +-1 regardless of |g|, so the
// parameter moves by lr*(1 + wd*p). Drop the correction and it moves by ~lr/10.
static bool test_bias_correction() {
    const int n = 256;
    const float lr = 0.01f, wd = 0.0f;
    std::vector<float> p(n, 0.5f), g(n, 3.0f), m(n, 0.0f), v(n, 0.0f);
    std::vector<float> gp = p;
    gpu_run(gp, g, m, v, lr, wd, 1);

    float worst = 0.0f;
    for (int i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs((p[i] - gp[i]) - lr));
    bool ok = worst <= 1e-6f;
    std::printf("  %-52s moved %.6f, expected %.6f  %s\n", "step 1 moves by lr (bias corrected)",
                p[0] - gp[0], lr, ok ? "PASS" : "FAIL");
    if (!ok)
        std::printf("      -> if it moved by about %.6f you dropped the 1/(1-beta^t) terms\n",
                    lr * (1.0f - B1));
    return ok;
}

// AdamW applies wd to p directly. Adam-with-L2 folds wd*p into g and lets it go
// through m and v. They must differ.
static bool test_decoupled() {
    const int n = 128;
    const float lr = 0.01f, wd = 0.1f;
    std::vector<float> p(n, 0.7f), g(n, 2.0f);

    std::vector<float> m1(n, 0.0f), v1(n, 0.0f), p1 = p;
    gpu_run(p1, g, m1, v1, lr, wd, 5);

    std::vector<float> g_l2(n);
    for (int i = 0; i < n; ++i) g_l2[i] = g[i] + wd * p[i];
    std::vector<float> m2(n, 0.0f), v2(n, 0.0f), p2 = p;
    gpu_run(p2, g_l2, m2, v2, lr, 0.0f, 5);

    float diff = 0.0f;
    for (int i = 0; i < n; ++i) diff = std::fmax(diff, std::fabs(p1[i] - p2[i]));
    bool ok = diff > 1e-4f;
    std::printf("  %-52s gap %.6f  %s\n", "AdamW differs from Adam+L2 (decoupled wd)", diff,
                ok ? "PASS" : "FAIL");
    if (!ok) std::printf("      -> they matched, so weight decay went through m and v\n");
    return ok;
}

// f(x) = sum(x^2), grad = 2x. Must fall monotonically to near zero.
static bool test_convergence(std::mt19937& rng) {
    const int n = 512;
    const float lr = 0.05f;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (float& q : x) q = dist(rng);

    std::vector<float> m(n, 0.0f), v(n, 0.0f), g(n);
    double prev = 1e30;
    bool monotonic = true;
    int nsteps = 200;

    Device_Buffer d_p(n), d_g(n), d_m(n), d_v(n);
    d_p.upload(x.data());
    d_m.upload(m.data());
    d_v.upload(v.data());
    for (int t = 1; t <= nsteps; ++t) {
        d_p.download(x.data());
        double loss = 0.0;
        for (int i = 0; i < n; ++i) {
            loss += (double)x[i] * (double)x[i];
            g[i] = 2.0f * x[i];
        }
        if (t > 1 && loss > prev + 1e-6) monotonic = false;
        prev = loss;
        d_g.upload(g.data());
        launch_adamw_step(d_p.ptr, d_g.ptr, d_m.ptr, d_v.ptr, n, lr, B1, B2, EPS, 0.0f, t);
    }
    d_p.download(x.data());
    double final_loss = 0.0;
    for (int i = 0; i < n; ++i) final_loss += (double)x[i] * (double)x[i];
    d_p.free_it();
    d_g.free_it();
    d_m.free_it();
    d_v.free_it();

    bool ok = monotonic && final_loss < 1e-6;
    std::printf("  %-52s final %.3e, monotonic %s  %s\n", "minimise sum(x^2) in 200 steps",
                final_loss, monotonic ? "yes" : "NO", ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    std::mt19937 rng(20260921);
    int failures = 0;

    std::printf("[single step vs CPU oracle]\n");
    for (float lr : {0.001f, 0.01f})
        for (float wd : {0.0f, 0.1f})
            if (!test_steps(4096, lr, wd, 1, rng)) ++failures;

    std::printf("[fifty steps vs CPU oracle]\n");
    for (float wd : {0.0f, 0.1f})
        if (!test_steps(4096, 0.001f, wd, 50, rng)) ++failures;

    std::printf("[bias correction]\n");
    if (!test_bias_correction()) ++failures;

    std::printf("[decoupled weight decay - the W in AdamW]\n");
    if (!test_decoupled()) ++failures;

    std::printf("[convergence]\n");
    if (!test_convergence(rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
