// ============================================================================
// Causal multi-head self-attention: correctness + gradient harness.
//
//   qkv (B,T,3C) -> att (B,NH,T,T) -> out (B,T,C)
//
// Checks:
//   1. forward vs CPU oracle
//   2. att rows sum to 1 over t2 <= t, and are exactly 0 for t2 > t
//   3. CAUSALITY PROBE: perturb qkv at position t2, assert out is bitwise
//      unchanged at every t < t2. Run this one first. A model that peeks one
//      token ahead trains beautifully and generates garbage, and no other test
//      here will catch it.
//   4. dqkv vs oracle
//   5. finite-difference gradient check on L = sum(dout * out) over qkv
//
// Shapes include a ragged non-power-of-two T on purpose: the causal mask
// transposes in the backward pass (a query at t reads keys at t2 <= t, so a key
// at t2 is read by queries at t >= t2) and getting that wrong is only visible
// at larger T.
//
// Exit 0 = all pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/attention.cuh"

// ---------------------------------------------------------------- CPU oracles

static void cpu_attn_fwd(const float* qkv, float* att, float* out, int B, int T, int C, int NH) {
    int HS = C / NH;
    double scale = 1.0 / std::sqrt((double)HS);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h)
            for (int t = 0; t < T; ++t) {
                const float* q = qkv + (size_t)b * T * 3 * C + (size_t)t * 3 * C + 0 * C + h * HS;
                float* arow = att + ((size_t)b * NH + h) * T * T + (size_t)t * T;

                std::vector<double> pre(t + 1);
                double m = -1e30;
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* k =
                        qkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 1 * C + h * HS;
                    double d = 0.0;
                    for (int i = 0; i < HS; ++i) d += (double)q[i] * (double)k[i];
                    pre[t2] = d * scale;
                    if (pre[t2] > m) m = pre[t2];
                }
                double s = 0.0;
                for (int t2 = 0; t2 <= t; ++t2) {
                    pre[t2] = std::exp(pre[t2] - m);
                    s += pre[t2];
                }
                for (int t2 = 0; t2 <= t; ++t2) arow[t2] = (float)(pre[t2] / s);
                for (int t2 = t + 1; t2 < T; ++t2) arow[t2] = 0.0f;

                float* o = out + (size_t)b * T * C + (size_t)t * C + h * HS;
                for (int i = 0; i < HS; ++i) {
                    double acc = 0.0;
                    for (int t2 = 0; t2 <= t; ++t2) {
                        const float* v =
                            qkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 2 * C + h * HS;
                        acc += (double)arow[t2] * (double)v[i];
                    }
                    o[i] = (float)acc;
                }
            }
}

static void cpu_attn_bwd(const float* qkv, const float* att, const float* dout, float* dqkv, int B,
                         int T, int C, int NH) {
    int HS = C / NH;
    double scale = 1.0 / std::sqrt((double)HS);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h) {
            for (int t = 0; t < T; ++t) {
                const float* arow = att + ((size_t)b * NH + h) * T * T + (size_t)t * T;
                const float* dO = dout + (size_t)b * T * C + (size_t)t * C + h * HS;

                std::vector<double> datt(t + 1, 0.0);
                for (int t2 = 0; t2 <= t; ++t2) {
                    const float* v =
                        qkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 2 * C + h * HS;
                    double d = 0.0;
                    for (int i = 0; i < HS; ++i) d += (double)dO[i] * (double)v[i];
                    datt[t2] = d;
                    float* dv = dqkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 2 * C + h * HS;
                    for (int i = 0; i < HS; ++i)
                        dv[i] += (float)((double)arow[t2] * (double)dO[i]);
                }

                double dot_pa = 0.0;
                for (int t2 = 0; t2 <= t; ++t2) dot_pa += (double)arow[t2] * datt[t2];

                for (int t2 = 0; t2 <= t; ++t2) {
                    double dpre = (double)arow[t2] * (datt[t2] - dot_pa);
                    const float* k =
                        qkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 1 * C + h * HS;
                    const float* q =
                        qkv + (size_t)b * T * 3 * C + (size_t)t * 3 * C + 0 * C + h * HS;
                    float* dq = dqkv + (size_t)b * T * 3 * C + (size_t)t * 3 * C + 0 * C + h * HS;
                    float* dk = dqkv + (size_t)b * T * 3 * C + (size_t)t2 * 3 * C + 1 * C + h * HS;
                    for (int i = 0; i < HS; ++i) {
                        dq[i] += (float)(scale * dpre * (double)k[i]);
                        dk[i] += (float)(scale * dpre * (double)q[i]);
                    }
                }
            }
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

static bool report(const char* label, int B, int T, int C, int NH, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  B=%-2d T=%-4d C=%-4d NH=%-3d %-26s err=%.2e  %s\n", B, T, C, NH, label, err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------- GPU helpers

static void gpu_forward(const std::vector<float>& qkv, std::vector<float>& att,
                        std::vector<float>& out, int B, int T, int C, int NH) {
    Device_Buffer d_qkv(B * T * 3 * C), d_att(B * NH * T * T),
        d_out(B * T * C);
    d_qkv.upload(const_cast<float*>(qkv.data()));
    launch_attention_fwd(d_qkv.ptr, d_att.ptr, d_out.ptr, B, T, C, NH);
    d_att.download(att.data());
    d_out.download(out.data());
    d_qkv.free_it();
    d_att.free_it();
    d_out.free_it();
}

static void gpu_backward(const std::vector<float>& qkv, const std::vector<float>& att,
                         const std::vector<float>& dout, std::vector<float>& dqkv, int B, int T,
                         int C, int NH) {
    std::vector<float> zero((size_t)B * T * 3 * C, 0.0f);
    Device_Buffer d_qkv(B * T * 3 * C), d_att(B * NH * T * T),
        d_dout(B * T * C);
    Device_Buffer d_datt(B * NH * T * T), d_dpre(B * NH * T * T),
        d_dqkv(B * T * 3 * C);
    d_qkv.upload(const_cast<float*>(qkv.data()));
    d_att.upload(const_cast<float*>(att.data()));
    d_dout.upload(const_cast<float*>(dout.data()));
    d_dqkv.upload(zero.data());
    launch_attention_bwd(d_qkv.ptr, d_att.ptr, d_dout.ptr, d_datt.ptr, d_dpre.ptr, d_dqkv.ptr, B, T,
                         C, NH);
    d_dqkv.download(dqkv.data());
    d_qkv.free_it();
    d_att.free_it();
    d_dout.free_it();
    d_datt.free_it();
    d_dpre.free_it();
    d_dqkv.free_it();
}

// ---------------------------------------------------------------- the tests

static bool test_causality(int B, int T, int C, int NH, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> qkv((size_t)B * T * 3 * C);
    for (float& v : qkv) v = dist(rng);

    std::vector<float> att((size_t)B * NH * T * T), out((size_t)B * T * C);
    gpu_forward(qkv, att, out, B, T, C, NH);

    int t2 = T / 2;
    std::vector<float> bumped = qkv;
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < 3 * C; ++j)
            bumped[(size_t)b * T * 3 * C + (size_t)t2 * 3 * C + j] += 7.0f;

    std::vector<float> att2((size_t)B * NH * T * T), out2((size_t)B * T * C);
    gpu_forward(bumped, att2, out2, B, T, C, NH);

    float leak = 0.0f;
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < t2; ++t)
            for (int c = 0; c < C; ++c) {
                size_t i = (size_t)b * T * C + (size_t)t * C + c;
                leak = std::fmax(leak, std::fabs(out[i] - out2[i]));
            }
    return report("no leak from the future", B, T, C, NH, leak, 0.0f);
}

static bool test_forward(int B, int T, int C, int NH, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> qkv((size_t)B * T * 3 * C);
    for (float& v : qkv) v = dist(rng);

    std::vector<float> att((size_t)B * NH * T * T), out((size_t)B * T * C);
    std::vector<float> ratt((size_t)B * NH * T * T), rout((size_t)B * T * C);
    gpu_forward(qkv, att, out, B, T, C, NH);
    cpu_attn_fwd(qkv.data(), ratt.data(), rout.data(), B, T, C, NH);

    bool ok = report("fwd  out", B, T, C, NH, worst_err(out, rout), 1e-4f);
    ok &= report("fwd  att", B, T, C, NH, worst_err(att, ratt), 1e-4f);

    float worst_sum = 0.0f, worst_mask = 0.0f;
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < NH; ++h)
            for (int t = 0; t < T; ++t) {
                const float* row = att.data() + ((size_t)b * NH + h) * T * T + (size_t)t * T;
                double s = 0.0;
                for (int j = 0; j <= t; ++j) s += row[j];
                worst_sum = std::fmax(worst_sum, (float)std::fabs(s - 1.0));
                for (int j = t + 1; j < T; ++j)
                    worst_mask = std::fmax(worst_mask, std::fabs(row[j]));
            }
    ok &= report("att rows sum to 1", B, T, C, NH, worst_sum, 1e-4f);
    ok &= report("att is 0 above diagonal", B, T, C, NH, worst_mask, 0.0f);
    return ok;
}

static bool test_backward(int B, int T, int C, int NH, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> qkv((size_t)B * T * 3 * C), dout((size_t)B * T * C);
    for (float& v : qkv) v = dist(rng);
    for (float& v : dout) v = dist(rng);

    std::vector<float> att((size_t)B * NH * T * T), out((size_t)B * T * C);
    gpu_forward(qkv, att, out, B, T, C, NH);

    std::vector<float> g((size_t)B * T * 3 * C), r((size_t)B * T * 3 * C, 0.0f);
    gpu_backward(qkv, att, dout, g, B, T, C, NH);
    cpu_attn_bwd(qkv.data(), att.data(), dout.data(), r.data(), B, T, C, NH);
    return report("bwd  dqkv", B, T, C, NH, worst_err(g, r), 2e-4f);
}

static bool test_finite_difference(int B, int T, int C, int NH, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> qkv((size_t)B * T * 3 * C), dout((size_t)B * T * C);
    for (float& v : qkv) v = dist(rng);
    for (float& v : dout) v = dist(rng);

    std::vector<float> att((size_t)B * NH * T * T), out((size_t)B * T * C);
    gpu_forward(qkv, att, out, B, T, C, NH);

    std::vector<float> g((size_t)B * T * 3 * C);
    gpu_backward(qkv, att, dout, g, B, T, C, NH);

    auto loss = [&]() {
        gpu_forward(qkv, att, out, B, T, C, NH);
        double s = 0.0;
        for (size_t i = 0; i < out.size(); ++i) s += (double)dout[i] * (double)out[i];
        return s;
    };

    const float h = 1e-2f;
    float worst = 0.0f;
    for (size_t j = 0; j < qkv.size(); ++j) {
        float saved = qkv[j];
        qkv[j] = saved + h;
        double lp = loss();
        qkv[j] = saved - h;
        double lm = loss();
        qkv[j] = saved;
        float fd = (float)((lp - lm) / (2.0 * (double)h));
        float e = std::fabs(g[j] - fd) / (std::fabs(fd) + 1.0f);
        if (e > worst) worst = e;
    }
    return report("grad check  dL/dqkv", B, T, C, NH, worst, 3e-3f);
}

int main() {
    std::mt19937 rng(777);
    int failures = 0;

    struct S {
        int B, T, C, NH;
    };
    S shapes[] = {
        {1, 4, 4, 1},    // tiny, single head
        {2, 6, 8, 2},    // two heads
        {2, 13, 12, 3},  // ragged T, three heads
        {2, 64, 384, 6}, // GPT-2 small config
    };

    std::printf("[causality probe - run this one first]\n");
    for (S s : shapes)
        if (!test_causality(s.B, s.T, s.C, s.NH, rng)) ++failures;

    std::printf("[attention forward]\n");
    for (S s : shapes)
        if (!test_forward(s.B, s.T, s.C, s.NH, rng)) ++failures;

    std::printf("[attention backward vs CPU oracle]\n");
    for (S s : shapes)
        if (!test_backward(s.B, s.T, s.C, s.NH, rng)) ++failures;

    std::printf("[gradient check vs finite difference, L = sum(dout * out)]\n");
    S small[] = {{1, 4, 4, 1}, {2, 6, 8, 2}};
    for (S s : small)
        if (!test_finite_difference(s.B, s.T, s.C, s.NH, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
