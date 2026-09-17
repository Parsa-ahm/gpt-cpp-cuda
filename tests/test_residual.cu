// ============================================================================
// Correctness harness for the residual add.
//
// The forward check is a formality. The three backward checks are the point:
//
//   2. backward from ZEROED grad buffers    -> d_a == d_b == dy
//   3. backward from JUNK-FILLED buffers    -> d_a == junk + dy   (accumulation)
//   4. backward with d_a and d_b ALIASED    -> d_a == junk + 2*dy (fan-out)
//
// A kernel that writes `d_a[i] = dy[i]` instead of `+=` passes 1 and 2 and
// fails 3 and 4. That bug does not crash and does not show up as a wrong
// number anywhere; it silently drops the skip-connection's gradient and you
// find out three days later when training will not descend. Hence test 3.
//
// Test 4 is the real usage: in a transformer the same tensor feeds both
// branches of the add, so the caller passes the same pointer twice and expects
// both contributions to land.
//
// Exit 0 = all checks pass.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/device_buffer.hpp"
#include "nn/residual.cuh"

static float max_abs_diff(const std::vector<float>& got, const std::vector<float>& ref) {
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float d = std::fabs(got[i] - ref[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static bool report(const char* label, int n, float err, float tol) {
    bool ok = err <= tol;
    std::printf("  n=%-6d  %-28s max|err|=%.2e  %s\n", n, label, err, ok ? "PASS" : "FAIL");
    return ok;
}

static bool test_forward(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> a(n), b(n), out(n), ref(n);
    for (float& v : a) v = dist(rng);
    for (float& v : b) v = dist(rng);
    for (int i = 0; i < n; ++i) ref[i] = a[i] + b[i];

    Device_Buffer d_a(n), d_b(n), d_out(n);
    d_a.upload(a.data());
    d_b.upload(b.data());
    launch_residual_fwd(d_a.ptr, d_b.ptr, d_out.ptr, n);
    d_out.download(out.data());
    d_a.free_it();
    d_b.free_it();
    d_out.free_it();

    return report("fwd  out = a + b", n, max_abs_diff(out, ref), 0.0f);
}

// Backward from zeroed buffers: the plain chain-rule result.
static bool test_backward_zeroed(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> dy(n), zero(n, 0.0f), da(n), db(n);
    for (float& v : dy) v = dist(rng);

    Device_Buffer d_dy(n), d_da(n), d_db(n);
    d_dy.upload(dy.data());
    d_da.upload(zero.data());
    d_db.upload(zero.data());
    launch_residual_bwd(d_dy.ptr, d_da.ptr, d_db.ptr, n);
    d_da.download(da.data());
    d_db.download(db.data());
    d_dy.free_it();
    d_da.free_it();
    d_db.free_it();

    bool ok = report("bwd  d_a = dy (zeroed)", n, max_abs_diff(da, dy), 0.0f);
    ok &= report("bwd  d_b = dy (zeroed)", n, max_abs_diff(db, dy), 0.0f);
    return ok;
}

// THE test: buffers arrive holding gradient from another op. Must accumulate.
static bool test_backward_accumulates(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> dy(n), junk_a(n), junk_b(n), da(n), db(n), ref_a(n), ref_b(n);
    for (float& v : dy) v = dist(rng);
    for (float& v : junk_a) v = dist(rng);
    for (float& v : junk_b) v = dist(rng);
    for (int i = 0; i < n; ++i) {
        ref_a[i] = junk_a[i] + dy[i];
        ref_b[i] = junk_b[i] + dy[i];
    }

    Device_Buffer d_dy(n), d_da(n), d_db(n);
    d_dy.upload(dy.data());
    d_da.upload(junk_a.data());
    d_db.upload(junk_b.data());
    launch_residual_bwd(d_dy.ptr, d_da.ptr, d_db.ptr, n);
    d_da.download(da.data());
    d_db.download(db.data());
    d_dy.free_it();
    d_da.free_it();
    d_db.free_it();

    bool ok = report("bwd  d_a += dy (accumulate)", n, max_abs_diff(da, ref_a), 1e-6f);
    ok &= report("bwd  d_b += dy (accumulate)", n, max_abs_diff(db, ref_b), 1e-6f);
    return ok;
}

// Real usage: one tensor feeds both branches, so both grads land in one buffer.
static bool test_backward_aliased(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> dy(n), junk(n), got(n), ref(n);
    for (float& v : dy) v = dist(rng);
    for (float& v : junk) v = dist(rng);
    for (int i = 0; i < n; ++i) ref[i] = junk[i] + 2.0f * dy[i];

    Device_Buffer d_dy(n), d_shared(n);
    d_dy.upload(dy.data());
    d_shared.upload(junk.data());
    // same pointer for both gradient outputs
    launch_residual_bwd(d_dy.ptr, d_shared.ptr, d_shared.ptr, n);
    d_shared.download(got.data());
    d_dy.free_it();
    d_shared.free_it();

    return report("bwd  aliased -> junk + 2*dy", n, max_abs_diff(got, ref), 1e-6f);
}

int main() {
    std::mt19937 rng(4242);
    int failures = 0;
    const int sizes[] = {1, 7, 255, 256, 257, 1000, 100000};

    std::printf("[residual forward]\n");
    for (int n : sizes)
        if (!test_forward(n, rng)) ++failures;

    std::printf("[residual backward, zeroed buffers]\n");
    for (int n : sizes)
        if (!test_backward_zeroed(n, rng)) ++failures;

    std::printf("[residual backward, ACCUMULATION into existing gradient]\n");
    for (int n : sizes)
        if (!test_backward_accumulates(n, rng)) ++failures;

    std::printf("[residual backward, ALIASED outputs (tensor feeds both branches)]\n");
    for (int n : sizes)
        if (!test_backward_aliased(n, rng)) ++failures;

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
