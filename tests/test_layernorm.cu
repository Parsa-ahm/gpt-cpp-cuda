// ============================================================================
// LayerNorm: correctness + gradient harness.
//
//   x (N,C) -> normalise each row -> * w[c] + b[c] -> out (N,C)
//
// Checks:
//   1. forward vs CPU oracle, plus mean and rstd
//   2. with w=1,b=0 every output row really has mean 0 and variance 1
//   3. dx vs oracle
//   4. dw, db vs oracle, ACCUMULATING into pre-filled buffers
//   5. finite-difference gradient check on L = sum(dout * out) over x, w, b
//
// Check 5 is what catches a dropped s1 or s2 term in the dx formula. Such a
// backward is wrong by a few percent everywhere and still looks plausible.
//
// Exit 0 = all pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/layernorm.cuh"

static const float LN_EPS = 1e-5f;

// ---------------------------------------------------------------- CPU oracles

static void cpu_ln_fwd(const float* x, const float* w, const float* b, float* out, float* mean,
                       float* rstd, int N, int C) {
    for (int n = 0; n < N; ++n) {
        double m = 0.0;
        for (int c = 0; c < C; ++c) m += (double)x[n * C + c];
        m /= C;
        double var = 0.0;
        for (int c = 0; c < C; ++c) {
            double d = (double)x[n * C + c] - m;
            var += d * d;
        }
        var /= C;
        double rs = 1.0 / std::sqrt(var + (double)LN_EPS);
        mean[n] = (float)m;
        rstd[n] = (float)rs;
        for (int c = 0; c < C; ++c) {
            double xhat = ((double)x[n * C + c] - m) * rs;
            out[n * C + c] = (float)(xhat * (double)w[c] + (double)b[c]);
        }
    }
}

static void cpu_ln_bwd(const float* dout, const float* x, const float* w, const float* mean,
                       const float* rstd, float* dx, float* dw, float* db, int N, int C) {
    std::vector<double> accw(C, 0.0), accb(C, 0.0);
    for (int n = 0; n < N; ++n) {
        double m = (double)mean[n], rs = (double)rstd[n];
        double s1 = 0.0, s2 = 0.0;
        for (int c = 0; c < C; ++c) {
            double xhat = ((double)x[n * C + c] - m) * rs;
            double dxhat = (double)dout[n * C + c] * (double)w[c];
            s1 += dxhat;
            s2 += dxhat * xhat;
        }
        s1 /= C;
        s2 /= C;
        for (int c = 0; c < C; ++c) {
            double xhat = ((double)x[n * C + c] - m) * rs;
            double dxhat = (double)dout[n * C + c] * (double)w[c];
            dx[n * C + c] = (float)(rs * (dxhat - s1 - xhat * s2));
            accw[c] += (double)dout[n * C + c] * xhat;
            accb[c] += (double)dout[n * C + c];
        }
    }
    for (int c = 0; c < C; ++c) {
        dw[c] += (float)accw[c];
        db[c] += (float)accb[c];
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

static bool report(const char* label, int N, int C, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  N=%-5d C=%-5d %-26s err=%.2e  %s\n", N, C, label, err, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------- GPU helpers

static void gpu_forward(const std::vector<float>& x, const std::vector<float>& w,
                        const std::vector<float>& b, std::vector<float>& out,
                        std::vector<float>& mean, std::vector<float>& rstd, int N, int C) {
    Device_Buffer dx(N * C), dw(C), db(C), dout(N * C), dmean(N), drstd(N);
    dx.upload(const_cast<float*>(x.data()));
    dw.upload(const_cast<float*>(w.data()));
    db.upload(const_cast<float*>(b.data()));
    launch_layernorm_fwd(dx.ptr, dw.ptr, db.ptr, dout.ptr, dmean.ptr, drstd.ptr, N, C);
    dout.download(out.data());
    dmean.download(mean.data());
    drstd.download(rstd.data());
    dx.free_it();
    dw.free_it();
    db.free_it();
    dout.free_it();
    dmean.free_it();
    drstd.free_it();
}

// ---------------------------------------------------------------- the tests

static bool test_forward(int N, int C, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> x(N * C), w(C), b(C);
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);
    for (float& v : b) v = dist(rng);

    std::vector<float> out(N * C), mean(N), rstd(N);
    std::vector<float> rout(N * C), rmean(N), rrstd(N);
    gpu_forward(x, w, b, out, mean, rstd, N, C);
    cpu_ln_fwd(x.data(), w.data(), b.data(), rout.data(), rmean.data(), rrstd.data(), N, C);

    bool ok = report("fwd  out", N, C, worst_err(out, rout), 1e-5f);
    ok &= report("fwd  mean", N, C, worst_err(mean, rmean), 1e-5f);
    ok &= report("fwd  rstd", N, C, worst_err(rstd, rrstd), 1e-5f);
    return ok;
}

// with w=1, b=0 the output rows must be standardised
static bool test_standardised(int N, int C, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> x(N * C), w(C, 1.0f), b(C, 0.0f);
    for (float& v : x) v = dist(rng);

    std::vector<float> out(N * C), mean(N), rstd(N);
    gpu_forward(x, w, b, out, mean, rstd, N, C);

    float worst_mean = 0.0f, worst_var = 0.0f;
    for (int n = 0; n < N; ++n) {
        double m = 0.0;
        for (int c = 0; c < C; ++c) m += (double)out[n * C + c];
        m /= C;
        double var = 0.0;
        for (int c = 0; c < C; ++c) {
            double d = (double)out[n * C + c] - m;
            var += d * d;
        }
        var /= C;
        worst_mean = std::fmax(worst_mean, (float)std::fabs(m));
        worst_var = std::fmax(worst_var, (float)std::fabs(var - 1.0));
    }
    bool ok = report("rows have mean 0", N, C, worst_mean, 1e-4f);
    ok &= report("rows have var 1", N, C, worst_var, 1e-3f);
    return ok;
}

static bool test_backward(int N, int C, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N * C), w(C), b(C), dout(N * C);
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);
    for (float& v : b) v = dist(rng);
    for (float& v : dout) v = dist(rng);

    std::vector<float> junk_dw(C), junk_db(C);
    for (float& v : junk_dw) v = dist(rng);
    for (float& v : junk_db) v = dist(rng);

    // forward first to produce the cached mean/rstd the backward needs
    std::vector<float> fout(N * C), mean(N), rstd(N);
    gpu_forward(x, w, b, fout, mean, rstd, N, C);

    std::vector<float> gx(N * C), gw(C), gb(C);
    Device_Buffer d_dout(N * C), d_x(N * C), d_w(C), d_mean(N), d_rstd(N);
    Device_Buffer d_dx(N * C), d_dw(C), d_db(C);
    d_dout.upload(dout.data());
    d_x.upload(x.data());
    d_w.upload(w.data());
    d_mean.upload(mean.data());
    d_rstd.upload(rstd.data());
    d_dw.upload(junk_dw.data());
    d_db.upload(junk_db.data());
    launch_layernorm_bwd(d_dout.ptr, d_x.ptr, d_w.ptr, d_mean.ptr, d_rstd.ptr, d_dx.ptr, d_dw.ptr,
                         d_db.ptr, N, C);
    d_dx.download(gx.data());
    d_dw.download(gw.data());
    d_db.download(gb.data());
    d_dout.free_it();
    d_x.free_it();
    d_w.free_it();
    d_mean.free_it();
    d_rstd.free_it();
    d_dx.free_it();
    d_dw.free_it();
    d_db.free_it();

    std::vector<float> rx(N * C), rw = junk_dw, rb = junk_db;
    cpu_ln_bwd(dout.data(), x.data(), w.data(), mean.data(), rstd.data(), rx.data(), rw.data(),
               rb.data(), N, C);

    bool ok = report("bwd  dx", N, C, worst_err(gx, rx), 1e-4f);
    ok &= report("bwd  dw +=", N, C, worst_err(gw, rw), 1e-4f);
    ok &= report("bwd  db +=", N, C, worst_err(gb, rb), 1e-4f);
    return ok;
}

static bool test_finite_difference(int N, int C, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N * C), w(C), b(C), dout(N * C);
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);
    for (float& v : b) v = dist(rng);
    for (float& v : dout) v = dist(rng);

    std::vector<float> fout(N * C), mean(N), rstd(N);
    gpu_forward(x, w, b, fout, mean, rstd, N, C);

    std::vector<float> gx(N * C), gw(C, 0.0f), gb(C, 0.0f);
    std::vector<float> zw(C, 0.0f), zb(C, 0.0f);
    {
        Device_Buffer d_dout(N * C), d_x(N * C), d_w(C), d_mean(N), d_rstd(N);
        Device_Buffer d_dx(N * C), d_dw(C), d_db(C);
        d_dout.upload(dout.data());
        d_x.upload(x.data());
        d_w.upload(w.data());
        d_mean.upload(mean.data());
        d_rstd.upload(rstd.data());
        d_dw.upload(zw.data());
        d_db.upload(zb.data());
        launch_layernorm_bwd(d_dout.ptr, d_x.ptr, d_w.ptr, d_mean.ptr, d_rstd.ptr, d_dx.ptr,
                             d_dw.ptr, d_db.ptr, N, C);
        d_dx.download(gx.data());
        d_dw.download(gw.data());
        d_db.download(gb.data());
        d_dout.free_it();
        d_x.free_it();
        d_w.free_it();
        d_mean.free_it();
        d_rstd.free_it();
        d_dx.free_it();
        d_dw.free_it();
        d_db.free_it();
    }

    std::vector<float> o(N * C), mm(N), rr(N);
    auto loss = [&]() {
        gpu_forward(x, w, b, o, mm, rr, N, C);
        double s = 0.0;
        for (int i = 0; i < N * C; ++i) s += (double)dout[i] * (double)o[i];
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
        return report(name, N, C, worst, 3e-3f);
    };

    bool ok = sweep(x, gx, "grad check  dL/dx");
    ok &= sweep(w, gw, "grad check  dL/dw");
    ok &= sweep(b, gb, "grad check  dL/db");
    return ok;
}

int main() {
    std::mt19937 rng(4242);
    int failures = 0;

    struct S {
        int N, C;
    };
    S shapes[] = {
        {4, 3},     // tiny
        {7, 5},     // no divisibility
        {256, 384}, // GPT-2 small width, one block of tokens
        {64, 768},  // GPT-2 124M width
        {129, 97},  // ragged
    };

    std::printf("[layernorm forward]\n");
    for (S s : shapes)
        if (!test_forward(s.N, s.C, rng)) ++failures;

    std::printf("[output rows are standardised when w=1, b=0]\n");
    for (S s : shapes)
        if (!test_standardised(s.N, s.C, rng)) ++failures;

    std::printf("[layernorm backward vs CPU oracle, accumulating into pre-filled grads]\n");
    for (S s : shapes)
        if (!test_backward(s.N, s.C, rng)) ++failures;

    std::printf("[gradient check vs finite difference, L = sum(dout * out)]\n");
    S small[] = {{4, 3}, {7, 5}};
    for (S s : small)
        if (!test_finite_difference(s.N, s.C, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
