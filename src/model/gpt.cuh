#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/device_buffer.hpp"
#include "rng/philox.hpp"

constexpr int NUM_PARAM_TENSORS = 16;
constexpr int NUM_ACT_TENSORS = 22;

// GPT config

struct GPT_Config {
    int max_seq_len;  // longest seq wpe can address
    int vocab_size;   // Token count, size of wte, V
    int n_layer;      // Number of transformer blocks, L
    int n_head;       // Number of attention heads, NH
    int n_embd;       // Stream width, C
};

// GPT params

struct GPT_Params {
    float* wte;       // (V, C) embedding table
    float* wpe;       // (maxT, C) possition embed
    float* ln1w;      // (L, C) layernorm 1 gain
    float* ln1b;      // (L, C) layernorm 1 bias
    float* qkvw;      // (L, C, 3C) fused Q/K/V projection weights
    float* qkvb;      // (L, 3C) its bias
    float* attprojw;  // (L, C, C) attention output projections
    float* attprojb;  // (L, C) its bias
    float* ln2w;      // (L, C) layernorm 2 gain
    float* ln2b;      // (L, C) its bias
    float* fcw;       // (L, C, 4C) MLP up-projection
    float* fcb;       // (L, 4C) its bias
    float* fcprojw;   // (L, 4C, C) MLP down-projection
    float* fcprojb;   // (L, C) its bias
    float* lnfw;      // (C) final layernorm
    float* lnfb;      // (C) its bias
};

// GPT Activations

struct GPT_Acts {
    float* encoded;   // (B,T,C)        embedding out, layer 0's ln1 input
    float* ln1;       // (L,B,T,C)      ln1 out, qkv projection input
    float* ln1_mean;  // (L,B,T)        cached for layernorm bwd
    float* ln1_rstd;  // (L,B,T)        cached for layernorm bwd
    float* qkv;       // (L,B,T,3C)     fused q,k,v
    float* att;       // (L,B,NH,T,T)   softmax probs, needed by softmax bwd
    float* atty;      // (L,B,T,C)      attention out, attproj input
    float* attproj;   // (L,B,T,C)      attention branch out
    float* res2;      // (L,B,T,C)      stream after attention joins
    float* ln2;       // (L,B,T,C)      ln2 out, fc input
    float* ln2_mean;  // (L,B,T)
    float* ln2_rstd;  // (L,B,T)
    float* fch;       // (L,B,T,4C)     PRE-gelu,  gelu_bwd's x
    float* fch_gelu;  // (L,B,T,4C)     POST-gelu, fcproj's input
    float* fcproj;    // (L,B,T,C)      mlp branch out
    float* res3;      // (L,B,T,C)      block out, next block's input
    float* lnf;       // (B,T,C)        final layernorm out
    float* lnf_mean;  // (B,T)
    float* lnf_rstd;  // (B,T)
    float* logits;    // (B,T,V)        head out. 823MB at B=16,T=256
    float* lse;       // (B,T)          log-sum-exp, crossentropy bwd needs it
    float* losses;    // (B,T)          per-token loss, forward averages these
};

inline void gpt_param_sizes(
    const GPT_Config& cfg,           // The address to the GPT_Config struct
    size_t sizes[NUM_PARAM_TENSORS]  // Out put
) {
    size_t v = cfg.vocab_size;
    size_t l = cfg.n_layer;
    size_t c = cfg.n_embd;

    sizes[0] = v * c;
    sizes[1] = cfg.max_seq_len * c;
    sizes[2] = l * c;
    sizes[3] = l * c;
    sizes[4] = l * c * 3 * c;
    sizes[5] = l * c * 3;
    sizes[6] = l * c * c;
    sizes[7] = l * c;
    sizes[8] = l * c;
    sizes[9] = l * c;
    sizes[10] = l * c * c * 4;
    sizes[11] = l * c * 4;
    sizes[12] = l * c * c * 4;
    sizes[13] = l * c;
    sizes[14] = c;
    sizes[15] = c;
}

inline void gpt_param_views(float* base, const size_t sizes[NUM_PARAM_TENSORS], GPT_Params& out) {
    size_t off = 0;
    out.wte = base + off;
    off += sizes[0];
    out.wpe = base + off;
    off += sizes[1];
    out.ln1w = base + off;
    off += sizes[2];
    out.ln1b = base + off;
    off += sizes[3];
    out.qkvw = base + off;
    off += sizes[4];
    out.qkvb = base + off;
    off += sizes[5];
    out.attprojw = base + off;
    off += sizes[6];
    out.attprojb = base + off;
    off += sizes[7];
    out.ln2w = base + off;
    off += sizes[8];
    out.ln2b = base + off;
    off += sizes[9];
    out.fcw = base + off;
    off += sizes[10];
    out.fcb = base + off;
    off += sizes[11];
    out.fcprojw = base + off;
    off += sizes[12];
    out.fcprojb = base + off;
    off += sizes[13];
    out.lnfw = base + off;
    off += sizes[14];
    out.lnfb = base + off;
    off += sizes[15];
}

inline void gpt_acts_size(
    const GPT_Config& cfg,
    int b,
    int t,
    size_t sizes[NUM_ACT_TENSORS]
) {
    size_t v = cfg.vocab_size;
    size_t c = cfg.n_embd;
    size_t nh = cfg.n_head;
    size_t bt = (size_t)b * t;
    size_t btc = (size_t)b * c * t;
    size_t l = cfg.n_layer;

    sizes[0] = btc;
    sizes[1] = l * btc;
    sizes[2] = l * bt;
    sizes[3] = l * bt;
    sizes[4] = l * 3 * btc;
    sizes[5] = l * b * nh * t * t;
    sizes[6] = l * btc;
    sizes[7] = l * btc;
    sizes[8] = l * btc;
    sizes[9] = l * btc;
    sizes[10] = l * bt;
    sizes[11] = l * bt;
    sizes[12] = l * 4 * btc;
    sizes[13] = l * 4 * btc;
    sizes[14] = l * btc;
    sizes[15] = l * btc;
    sizes[16] = btc;
    sizes[17] = bt;
    sizes[18] = bt;
    sizes[19] = bt * v;
    sizes[20] = bt;
    sizes[21] = bt;
}

inline void gpt_act_views(float* base, const size_t sizes[NUM_ACT_TENSORS], GPT_Acts& out) {
    size_t off = 0;
    out.encoded = base + off;
    off += sizes[0];
    out.ln1 = base + off;
    off += sizes[1];
    out.ln1_mean = base + off;
    off += sizes[2];
    out.ln1_rstd = base + off;
    off += sizes[3];
    out.qkv = base + off;
    off += sizes[4];
    out.att = base + off;
    off += sizes[5];
    out.atty = base + off;
    off += sizes[6];
    out.attproj = base + off;
    off += sizes[7];
    out.res2 = base + off;
    off += sizes[8];
    out.ln2 = base + off;
    off += sizes[9];
    out.ln2_mean = base + off;
    off += sizes[10];
    out.ln2_rstd = base + off;
    off += sizes[11];
    out.fch = base + off;
    off += sizes[12];
    out.fch_gelu = base + off;
    off += sizes[13];
    out.fcproj = base + off;
    off += sizes[14];
    out.res3 = base + off;
    off += sizes[15];
    out.lnf = base + off;
    off += sizes[16];
    out.lnf_mean = base + off;
    off += sizes[17];
    out.lnf_rstd = base + off;
    off += sizes[18];
    out.logits = base + off;
    off += sizes[19];
    out.lse = base + off;
    off += sizes[20];
    out.losses = base + off;
    off += sizes[21];
}


struct GPT {
    GPT_Config cfg;
    int B, T;

    // owners
    Device_Buffer params_buf, grads_buf, m_buf, v_buf, acts_buf;

    // Aliases for the bufs
    float* params;
    float* grads;

    // Param size
    size_t n_params;
    size_t n_acts;

    GPT_Params p, g;
    GPT_Acts acts;

    void build(const GPT_Config& cfg, int B, int T);
    void init_random(uint64_t seed);
    void free_it();
};

inline void GPT::build(const GPT_Config& cfg, int B, int T) {
    this->cfg = cfg;
    this->B = B;
    this->T = T;

    //Checks
    assert(cfg.n_embd % cfg.n_head == 0);
    assert(T <= cfg.max_seq_len);

    size_t sp[NUM_PARAM_TENSORS];
    n_params = 0;
    gpt_param_sizes(cfg, sp);
    for (size_t s : sp) {
        n_params += s;
    }

    n_acts = 0;
    size_t sa[NUM_ACT_TENSORS];
    gpt_acts_size(cfg, B, T, sa);
    for (size_t s : sa) {
        n_acts += s;
    }

    params_buf.make(n_params);
    grads_buf.make(n_params);
    grads_buf.zero();
    m_buf.make(n_params);
    m_buf.zero();
    v_buf.make(n_params);
    v_buf.zero();
    acts_buf.make(n_acts);
    acts_buf.zero();

    // Aliases first: the views below are computed from them.
    params = params_buf.ptr;
    grads = grads_buf.ptr;

    // Same sp[] both times: that is what makes p and g identical in geometry.
    gpt_param_views(params, sp, p);
    gpt_param_views(grads, sp, g);
    gpt_act_views(acts_buf.ptr, sa, acts);

    std::printf("params: %zu (%.1f MB)\n", n_params, n_params * 4 / 1e6);
    std::printf("acts:   %zu (%.0f MB)\n", n_acts, n_acts * 4 / 1e6);
}

inline void GPT::init_random(uint64_t seed) {
    size_t sizes[NUM_PARAM_TENSORS];
    gpt_param_sizes(cfg, sizes);

    std::vector<float> h(n_params);

    uint32_t ctr[4] = {0, 0, 0, 0};
    uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    PhiloxStream rng;
    rng.init(ctr, key);

    const float sd_w = 0.02f;
    const float sd_proj = 0.02f / std::sqrt(2.0f * (float)cfg.n_layer);

    size_t off = 0;
    for (int i = 0; i < NUM_PARAM_TENSORS; i++) {
        float sd = 0.0f;     // > 0 means "normal with this std"
        float konst = 0.0f;  // used when sd == 0
        switch (i) {
            case 2:
            case 8:
            case 14:  // ln1w, ln2w, lnfw: gain of 1 passes the stream through
                konst = 1.0f;
                break;
            case 3:
            case 5:
            case 7:
            case 9:
            case 11:
            case 13:
            case 15:  // every bias
                konst = 0.0f;
                break;
            case 6:
            case 12:  // attprojw, fcprojw
                sd = sd_proj;
                break;
            default:  // wte, wpe, qkvw, fcw
                sd = sd_w;
                break;
        }
        for (size_t j = 0; j < sizes[i]; j++) {
            h[off + j] = (sd > 0.0f) ? rng.next_n() * sd : konst;
        }
        off += sizes[i];
    }
    assert(off == n_params);

    params_buf.upload(h.data());
}

inline void GPT::free_it() {
    params_buf.free_it();
    grads_buf.free_it();
    m_buf.free_it();
    v_buf.free_it();
    acts_buf.free_it();
}
