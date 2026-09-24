// ============================================================================
// Correctness + gradient harness for the linear layer.
//
//   x (N,C) @ W (C,OC) + b (OC)  ->  out (N,OC)
//
// Checks:
//   1. forward vs CPU oracle (incl. bias)
//   2. dx vs oracle
//   3. dW vs oracle, and that it ACCUMULATES into a pre-filled buffer
//   4. db vs oracle, and that it ACCUMULATES
//   5. finite-difference gradient check on the scalar loss L = sum(dy * out),
//      perturbing every element of x, W and b through the FORWARD kernels only
//
// Check 5 is the one that catches a swapped transpose. Checks 2-4 compare
// against an oracle written from the same formulas as the spec, so a shared
// misunderstanding survives them. The finite difference never touches the
// backward code, so it cannot.
//
// Exit 0 = all pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/linear.cuh"

// ---------------------------------------------------------------- CPU oracles

static void cpu_linear_fwd(const float* x, const float* W, const float* b, float* out, int N,
                           int C, int OC) {
    for (int n = 0; n < N; ++n)
        for (int o = 0; o < OC; ++o) {
            double sum = b ? (double)b[o] : 0.0;
            for (int c = 0; c < C; ++c) sum += (double)x[n * C + c] * (double)W[c * OC + o];
            out[n * OC + o] = (float)sum;
        }
}

static void cpu_linear_bwd(const float* x, const float* W, const float* dy, float* dx, float* dW,
                           float* db, int N, int C, int OC) {
    // dx = dy @ W^T   (assign)
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            double sum = 0.0;
            for (int o = 0; o < OC; ++o) sum += (double)dy[n * OC + o] * (double)W[c * OC + o];
            dx[n * C + c] = (float)sum;
        }
    // dW = x^T @ dy   (accumulate)
    for (int c = 0; c < C; ++c)
        for (int o = 0; o < OC; ++o) {
            double sum = 0.0;
            for (int n = 0; n < N; ++n) sum += (double)x[n * C + c] * (double)dy[n * OC + o];
            dW[c * OC + o] += (float)sum;
        }
    // db = column sums of dy   (accumulate)
    for (int o = 0; o < OC; ++o) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) sum += (double)dy[n * OC + o];
        db[o] += (float)sum;
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

static bool report(const char* label, int N, int C, int OC, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  N=%-4d C=%-4d OC=%-4d  %-26s err=%.2e  %s\n", N, C, OC, label, err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------- GPU helpers

static void gpu_forward(const std::vector<float>& x, const std::vector<float>& W,
                        const std::vector<float>& b, std::vector<float>& out, int N, int C,
                        int OC) {
    Device_Buffer dx(N * C), dW(C * OC), db(OC), dout(N * OC);
    dx.upload(const_cast<float*>(x.data()));
    dW.upload(const_cast<float*>(W.data()));
    db.upload(const_cast<float*>(b.data()));
    launch_linear_fwd(dx.ptr, dW.ptr, db.ptr, dout.ptr, N, C, OC);
    dout.download(out.data());
    dx.free_it();
    dW.free_it();
    db.free_it();
    dout.free_it();
}

// ---------------------------------------------------------------- the tests

static bool test_forward(int N, int C, int OC, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N * C), W(C * OC), b(OC), out(N * OC), ref(N * OC);
    for (float& v : x) v = dist(rng);
    for (float& v : W) v = dist(rng);
    for (float& v : b) v = dist(rng);

    gpu_forward(x, W, b, out, N, C, OC);
    cpu_linear_fwd(x.data(), W.data(), b.data(), ref.data(), N, C, OC);
    return report("fwd  x@W + b", N, C, OC, worst_err(out, ref), 1e-5f);
}

// backward, with pre-filled dW and db so accumulation is exercised
static bool test_backward(int N, int C, int OC, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N * C), W(C * OC), dy(N * OC);
    std::vector<float> junk_dW(C * OC), junk_db(OC);
    for (float& v : x) v = dist(rng);
    for (float& v : W) v = dist(rng);
    for (float& v : dy) v = dist(rng);
    for (float& v : junk_dW) v = dist(rng);
    for (float& v : junk_db) v = dist(rng);

    std::vector<float> gx(N * C), gW(C * OC), gb(OC);
    std::vector<float> rx(N * C), rW = junk_dW, rb = junk_db;

    Device_Buffer d_x(N * C), d_W(C * OC), d_dy(N * OC), d_dx(N * C), d_dW(C * OC), d_db(OC);
    d_x.upload(x.data());
    d_W.upload(W.data());
    d_dy.upload(dy.data());
    d_dW.upload(junk_dW.data());
    d_db.upload(junk_db.data());
    launch_linear_bwd(d_x.ptr, d_W.ptr, d_dy.ptr, d_dx.ptr, d_dW.ptr, d_db.ptr, N, C, OC);
    d_dx.download(gx.data());
    d_dW.download(gW.data());
    d_db.download(gb.data());
    d_x.free_it();
    d_W.free_it();
    d_dy.free_it();
    d_dx.free_it();
    d_dW.free_it();
    d_db.free_it();

    cpu_linear_bwd(x.data(), W.data(), dy.data(), rx.data(), rW.data(), rb.data(), N, C, OC);

    bool ok = report("bwd  dx = dy@W^T", N, C, OC, worst_err(gx, rx), 1e-5f);
    ok &= report("bwd  dW += x^T@dy", N, C, OC, worst_err(gW, rW), 1e-5f);
    ok &= report("bwd  db += colsum(dy)", N, C, OC, worst_err(gb, rb), 1e-5f);
    return ok;
}

// L = sum(dy * out), dy fixed. dL/dparam must equal what backward reported.
static bool test_finite_difference(int N, int C, int OC, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(N * C), W(C * OC), b(OC), dy(N * OC);
    for (float& v : x) v = dist(rng);
    for (float& v : W) v = dist(rng);
    for (float& v : b) v = dist(rng);
    for (float& v : dy) v = dist(rng);

    // analytic, from the kernels, into zeroed buffers
    std::vector<float> gx(N * C), gW(C * OC), gb(OC);
    std::vector<float> zW(C * OC, 0.0f), zb(OC, 0.0f);
    {
        Device_Buffer d_x(N * C), d_W(C * OC), d_dy(N * OC), d_dx(N * C), d_dW(C * OC), d_db(OC);
        d_x.upload(x.data());
        d_W.upload(W.data());
        d_dy.upload(dy.data());
        d_dW.upload(zW.data());
        d_db.upload(zb.data());
        launch_linear_bwd(d_x.ptr, d_W.ptr, d_dy.ptr, d_dx.ptr, d_dW.ptr, d_db.ptr, N, C, OC);
        d_dx.download(gx.data());
        d_dW.download(gW.data());
        d_db.download(gb.data());
        d_x.free_it();
        d_W.free_it();
        d_dy.free_it();
        d_dx.free_it();
        d_dW.free_it();
        d_db.free_it();
    }

    std::vector<float> out(N * OC);
    auto loss = [&](std::vector<float>& xv, std::vector<float>& Wv, std::vector<float>& bv) {
        gpu_forward(xv, Wv, bv, out, N, C, OC);
        double s = 0.0;
        for (int i = 0; i < N * OC; ++i) s += (double)dy[i] * (double)out[i];
        return s;
    };

    const float h = 1e-2f;
    auto sweep = [&](std::vector<float>& p, const std::vector<float>& analytic, const char* name) {
        float worst = 0.0f;
        for (size_t j = 0; j < p.size(); ++j) {
            float saved = p[j];
            p[j] = saved + h;
            double lp = loss(x, W, b);
            p[j] = saved - h;
            double lm = loss(x, W, b);
            p[j] = saved;
            float fd = (float)((lp - lm) / (2.0 * (double)h));
            float e = std::fabs(analytic[j] - fd) / (std::fabs(fd) + 1.0f);
            if (e > worst) worst = e;
        }
        return report(name, N, C, OC, worst, 2e-3f);
    };

    bool ok = sweep(x, gx, "grad check  dL/dx");
    ok &= sweep(W, gW, "grad check  dL/dW");
    ok &= sweep(b, gb, "grad check  dL/db");
    return ok;
}

int main() {
    std::mt19937 rng(31337);
    int failures = 0;

    struct S { int N, C, OC; };
    S shapes[] = {
        {4, 3, 2},       // tiny
        {7, 5, 3},       // no divisibility
        {64, 384, 1536}, // GPT-2 small MLP expand, one block of tokens
        {256, 384, 384}, // square-ish projection
        {129, 97, 61},   // ragged
    };

    std::printf("[linear forward]\n");
    for (S s : shapes)
        if (!test_forward(s.N, s.C, s.OC, rng)) ++failures;

    std::printf("[linear backward vs CPU oracle, accumulating into pre-filled grads]\n");
    for (S s : shapes)
        if (!test_backward(s.N, s.C, s.OC, rng)) ++failures;

    // Finite difference reruns the forward pass twice per parameter, so keep it
    // to small shapes. Every element gets perturbed.
    std::printf("[gradient check vs finite difference, L = sum(dy * out)]\n");
    S small[] = {{4, 3, 2}, {7, 5, 3}};
    for (S s : small)
        if (!test_finite_difference(s.N, s.C, s.OC, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
