// ============================================================================
// Correctness + gradient harness for GELU (tanh approximation).
//
// Three checks, in increasing strength:
//
//   1. FORWARD vs CPU oracle. Same formula, computed on the host in float.
//      Catches indexing bugs, missing bounds guards, wrong constants.
//
//   2. BACKWARD vs CPU oracle of the ANALYTIC derivative. Catches the same
//      class of bug in the backward kernel.
//
//   3. BACKWARD vs CENTRAL FINITE DIFFERENCE of the GPU's own forward kernel:
//          dx_i ~= dy_i * (gelu(x_i + h) - gelu(x_i - h)) / (2h)
//      This is the real gradient check. Checks 1 and 2 can both pass while the
//      derivative is wrong, if the same wrong formula was typed into the kernel
//      and the oracle. The finite difference only ever touches the FORWARD
//      pass, so it cannot share that mistake. Every op in Rung 2 gets this
//      treatment; this is the harness they will reuse.
//
//      The tolerance here is looser on purpose. Central differences in FP32
//      carry truncation error ~h^2 and cancellation error ~eps/h; at h=1e-2
//      that floors out around 1e-4 relative. A 1e-3-ish disagreement is the
//      method, not the kernel. A 1e-1 disagreement is the kernel.
//
// Sizes include non-multiples of 256 so the grid overshoots and the bounds
// guard is exercised.
//
// Exit 0 = all checks pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/gelu.cuh"

static const double kS = 0.7978845608028654;  // sqrt(2/pi)
static const double kC = 0.044715;

// CPU oracle in DOUBLE, deliberately. The kernel is FP32 and we want the
// reference to be better than the thing under test, not equally wrong.
static float cpu_gelu(float x) {
    double xd = (double)x;
    double inner = kS * (xd + kC * xd * xd * xd);
    return (float)(0.5 * xd * (1.0 + std::tanh(inner)));
}

// CPU oracle: d/dx gelu(x), analytic. Also double.
static float cpu_gelu_prime(float x) {
    double xd = (double)x;
    double inner = kS * (xd + kC * xd * xd * xd);
    double t = std::tanh(inner);
    return (float)(0.5 * (1.0 + t) +
                   0.5 * xd * (1.0 - t * t) * kS * (1.0 + 3.0 * kC * xd * xd));
}

// Error measure: absolute error, softened by the magnitude of the reference.
//
// The +1.0f denominator (not +1e-3f) is deliberate, and the reason is worth
// knowing. For x around -4, inner is about -6.9 and tanh(inner) is about
// -0.99997, so the forward pass computes 1.0f + (-0.99997f) and CATASTROPHIC
// CANCELLATION eats the precision: the result is ~3e-5 but carries the ~6e-8
// absolute error of a float near 1.0, i.e. ~0.2% RELATIVE error. That is not a
// kernel bug, it is what FP32 can represent, and no correct implementation does
// better. A pure relative measure would flag every correct kernel on the
// negative tail. Absolute error is the honest metric here: gelu's output range
// over the test inputs is about [-0.17, 4], so a 1e-5 bound still catches any
// wrong constant, missing term, or bad index by several orders of magnitude.
static float mixed_err(float got, float ref) {
    return std::fabs(got - ref) / (std::fabs(ref) + 1.0f);
}

// A spread of inputs: random over [-4,4] plus hand-picked edges (exact zero,
// deep saturation on both sides, the kink region around 0).
static void fill_inputs(std::vector<float>& x, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (float& v : x) v = dist(rng);
    const float edges[] = {0.0f, 1e-6f, -1e-6f, 6.0f, -6.0f, 0.5f, -0.5f, 3.0f, -3.0f};
    int m = (int)x.size() < 9 ? (int)x.size() : 9;
    for (int i = 0; i < m; ++i) x[i] = edges[i];
}

static bool test_forward(int n, std::mt19937& rng) {
    std::vector<float> x(n), y(n), ref(n);
    fill_inputs(x, rng);

    Device_Buffer dx(n), dy(n);
    dx.upload(x.data());
    launch_gelu_fwd(dx.ptr, dy.ptr, n);
    dy.download(y.data());
    dx.free_it();
    dy.free_it();

    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        ref[i] = cpu_gelu(x[i]);
        float e = mixed_err(y[i], ref[i]);
        if (e > worst) worst = e;
    }
    bool ok = worst < 1e-5f;
    std::printf("  n=%-6d  fwd   max rel err=%.2e  %s\n", n, worst, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_backward_analytic(int n, std::mt19937& rng) {
    std::vector<float> x(n), dy(n), dx(n);
    fill_inputs(x, rng);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : dy) v = dist(rng);

    Device_Buffer d_x(n), d_dy(n), d_dx(n);
    d_x.upload(x.data());
    d_dy.upload(dy.data());
    launch_gelu_bwd(d_x.ptr, d_dy.ptr, d_dx.ptr, n);
    d_dx.download(dx.data());
    d_x.free_it();
    d_dy.free_it();
    d_dx.free_it();

    float worst = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = dy[i] * cpu_gelu_prime(x[i]);
        float e = mixed_err(dx[i], ref);
        if (e > worst) worst = e;
    }
    bool ok = worst < 1e-5f;
    std::printf("  n=%-6d  bwd   max rel err=%.2e  %s   (vs analytic oracle)\n", n, worst,
                ok ? "PASS" : "FAIL");
    return ok;
}

// THE gradient check: analytic backward vs central difference of the GPU forward.
static bool test_backward_finite_difference(int n, std::mt19937& rng) {
    const float h = 1e-2f;
    std::vector<float> x(n), xp(n), xm(n), dy(n), dx(n), yp(n), ym(n);
    fill_inputs(x, rng);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : dy) v = dist(rng);
    for (int i = 0; i < n; ++i) {
        xp[i] = x[i] + h;
        xm[i] = x[i] - h;
    }

    Device_Buffer d_x(n), d_dy(n), d_dx(n), d_in(n), d_out(n);
    d_x.upload(x.data());
    d_dy.upload(dy.data());
    launch_gelu_bwd(d_x.ptr, d_dy.ptr, d_dx.ptr, n);
    d_dx.download(dx.data());

    d_in.upload(xp.data());
    launch_gelu_fwd(d_in.ptr, d_out.ptr, n);
    d_out.download(yp.data());

    d_in.upload(xm.data());
    launch_gelu_fwd(d_in.ptr, d_out.ptr, n);
    d_out.download(ym.data());

    d_x.free_it();
    d_dy.free_it();
    d_dx.free_it();
    d_in.free_it();
    d_out.free_it();

    float worst = 0.0f;
    int worst_i = 0;
    for (int i = 0; i < n; ++i) {
        float fd = dy[i] * (yp[i] - ym[i]) / (2.0f * h);
        float e = mixed_err(dx[i], fd);
        if (e > worst) {
            worst = e;
            worst_i = i;
        }
    }
    bool ok = worst < 2e-3f;
    std::printf("  n=%-6d  bwd   max rel err=%.2e  %s   (vs finite diff, h=%.0e, worst at x=%.3f)\n",
                n, worst, ok ? "PASS" : "FAIL", h, x[worst_i]);
    return ok;
}

int main() {
    std::mt19937 rng(12345);
    int failures = 0;

    // 1, 7, 255, 257, 1000, 5000 are not multiples of 256 -> grid overshoots.
    const int sizes[] = {1, 7, 255, 256, 257, 1000, 4096, 5000, 100000};

    std::printf("[gelu forward vs CPU oracle]\n");
    for (int n : sizes)
        if (!test_forward(n, rng)) ++failures;

    std::printf("[gelu backward vs CPU oracle]\n");
    for (int n : sizes)
        if (!test_backward_analytic(n, rng)) ++failures;

    std::printf("[gelu gradient check: backward vs central finite difference]\n");
    for (int n : sizes)
        if (!test_backward_finite_difference(n, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
