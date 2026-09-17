# gpt-cpp-cuda — Project Charter

**Status:** ACTIVE — Rungs 0-1 done (Philox PRNG, SGEMM ladder). Rung 2 (transformer ops fwd+bwd)
starting 2026-09-17. This document is the single source of truth for *why* this project exists and
*what done means*. If work drifts from this, either the work is wrong or this doc is wrong —
reconcile before continuing. Do not lose the plot.

> Pivoted 2026-09-16 from `image-inpainting-cpp-cuda` (score-based inpainting). Rungs 0-1 carry
> over unchanged. Reason: "trained an LLM in C++/CUDA" is legible to every recruiter; the
> inpainting story needed a paragraph. Score-matching math stays banked in `pml-notes`.

---

## 0. The one sentence

**Train a GPT from scratch in C++ on the raw CUDA stack (cuBLAS + hand-written kernels, no
PyTorch/JAX), forward and backward, run real GPT-2 124M weights through the same engine, and
benchmark the hand-written kernels against the library.**

## 1. Why this project exists

Capstone of the straight-ML phase, before the agents/RAG chapter. It must erase reasonable doubt
that Parsa is a real ML engineer by proving three things together:

1. **Depth** — understands the transformer from the matmul up, not via `nn.Module`.
2. **GPU systems** — has personally written and profiled the CUDA kernels for a training loop,
   knows how to drive cuBLAS from C++, and can explain where the bits move.
3. **LLM literacy** — GPT-2 architecture, training dynamics, inference, and the modern
   extensions (RoPE, MoE, SwiGLU, GQA, flash attention) in context.

Extends: `ml-jax-pytorch` (MoE study, ragged_dot kernel) and `pml-notes` (Murphy from scratch).

## 2. Thesis (the defensible claim)

> "I built a GPT training + inference engine in C++/CUDA with no ML framework: hand-written
> forward and backward kernels for every op except the matmul, which calls cuBLAS and which I also
> wrote myself and benchmarked against it. I trained a small GPT on my own GPU, loaded OpenAI's
> GPT-2 124M weights into my engine and matched HuggingFace logits, and wrote a fused attention
> kernel that I profiled against the naive version. I can explain and profile every line."

Capability + understanding + a working artifact. Not a quality benchmark. Never a comparative
thesis we cannot support (the MoE lesson).

## 3. Non-goals

- NOT training GPT-2 124M to convergence on a 2070S. We train a ~10M model on a small corpus and
  *run* 124M with loaded weights.
- NOT beating cuBLAS or flash-attention. We benchmark against them and document the gap.
- NOT a general framework. One model family (decoder-only GPT), one engine.
- NOT mixed precision / multi-GPU / distributed. FP32, one GPU. Named as future work.

## 4. Boundaries (the from-scratch line)

**The line is: no ML framework. Not: no CUDA libraries.**

- **Forbidden in the engine:** PyTorch, JAX, TensorFlow, any autograd/tensor library.
- **Allowed:** CUDA runtime, **cuBLAS** for matmuls in the model path, C++17 stdlib, CMake.
  cuDNN is installed but not needed; may be used for a softmax/layernorm *benchmark reference*.
- **Hand-written by Parsa:** every non-matmul kernel fwd+bwd (embedding, layernorm, GELU, softmax /
  causal attention, residual, cross-entropy), Adam, the autodiff composition (which call feeds
  which, what is cached for backward), the training loop, the tokenizer loader, the weight loader,
  the sampler. Plus the Rung 1 SGEMM ladder, kept as the standalone "I can write the kernel" flex.
- **Python allowed only for:** data prep (download corpus, tokenize with `tiktoken` to a binary
  blob), exporting HF GPT-2 weights to a flat binary, producing the reference logits for the
  Rung 4 check, and plotting. Never in the engine.

**Stack:** C++17 + CUDA 12.0 + cuBLAS 12. RTX 2070 SUPER (`sm_75`, 8 GB, 40 SMs). CMake ≥ 3.24,
host compiler **g++-12**.

## 5. Collaboration model (STRICT, unchanged)

**Parsa writes 100% of the engine.** Claude does not show implementation code for Parsa's parts.
Claude: explains the concept and math, writes precise task specs (inputs/outputs/shapes/invariants,
never the code), writes the tests + CPU reference oracle, handles build/data-prep/plotting/review,
and pair-optimizes only *after* a kernel is correct. Teaching rule: when Parsa writes non-optimal
CUDA, explain why on both axes (GPU architecture reason + C++ language reason).

Pace rule: one step at a time, short concrete walkthroughs. Reference implementations (llm.c) are
off-limits until Parsa's version of that piece passes its test.

## 6. The model (spec)

Decoder-only GPT-2 architecture, exactly as Karpathy's "Let's reproduce GPT-2":
token embedding + learned positional embedding → N × [LayerNorm → causal multi-head attention →
residual → LayerNorm → MLP (Linear 4x, GELU, Linear) → residual] → final LayerNorm → tied
LM head → cross-entropy. AdamW. Cosine LR with warmup.

- **Rung 3 training config:** ~10M params (e.g. n_layer 6, n_head 6, n_embd 384, ctx 256),
  TinyShakespeare (char or GPT-2 BPE) or TinyStories subset. Fits 8 GB FP32 comfortably.
- **Rung 4 inference config:** GPT-2 124M (12 layers, 12 heads, 768, ctx 1024, vocab 50257),
  weights loaded from a flat binary exported from HuggingFace.

## 7. Build ladder (each rung shippable)

- **Rung 0 — Scaffold + PRNG.** ✓ DONE. CMake/CUDA build, GPU sanity, hand-written Philox 4x32-10
  (KAT-exact vs Random123), Box-Muller normals. Tests green.
- **Rung 1 — SGEMM ladder.** ✓ DONE. naive → 16x16 tiled → 64x64 register-blocked (8x8
  micro-tile). ~2200 GFLOP/s @ 1024, ~40% cuBLAS on 2070S. `Device_Buffer` RAII-ish wrapper.
- **Rung 2 — Ops, fwd + bwd, gradient-checked.** Each op = forward kernel + backward kernel +
  Claude's CPU oracle + finite-difference gradient check. Order: GELU (on-ramp) → residual/add →
  linear via cuBLAS (learn row-major vs column-major, the transpose trick) → layernorm → embedding
  + positional → causal attention (QKV projection, batched QK^T via cuBLAS, custom masked softmax,
  AV via cuBLAS; backward through all of it) → cross-entropy (fused with softmax) → AdamW update.
  *Artifact:* every op passes gradient check at 1e-4 rel. **← the hard rung.**
- **Rung 3 — Train.** Compose the ops into the GPT forward, the backward in reverse, the training
  loop, LR schedule, checkpointing. Train ~10M on TinyShakespeare. *Artifact:* loss curve to ~1.5
  (char) / samples that read as Shakespeare-ish. **← ML floor.**
- **Rung 4 — Real weights.** Python exports HF GPT-2 124M to flat binary + reference logits for a
  fixed prompt. Engine loads, runs forward, matches logits (max abs diff < 1e-3 FP32), samples
  text with top-k. *Artifact:* coherent GPT-2 text from our engine. **← the "it's real" proof.**
- **Rung 5 — Optimize + benchmark.** Fused attention kernel (one kernel: QK^T, mask, softmax, AV
  in shared memory / registers) vs the naive 3-call version; SGEMM vs cuBLAS chart; per-kernel
  timings; roofline. *Artifact:* charts + numbers + explanation of the gap. **← systems flex.**
- **Rung 6 — Writeup + publish.** README with architecture diagram, math derivations of every
  backward pass, benchmark charts, honest limitations. Public repo. One post.

## 8. Metrics

- Per-kernel: time, GFLOP/s or GB/s, % of cuBLAS (matmul) / % of naive (attention), roofline.
- Training: loss curve, tokens/sec, step time, GPU memory.
- Correctness: gradient-check max rel error per op; Rung 4 logit max abs diff vs HF.
- Inference: tokens/sec for 124M generation.

## 9. Success criteria (definition of done)

- Rungs 2-6 artifacts reproducible from a clean clone with documented commands.
- Parsa can, cold: derive any backward pass on a whiteboard, explain every kernel's memory
  pattern, explain the GPT-2 architecture and its modern successors.
- Public repo + writeup a senior ML engineer would accept as real.
- **Timebox: Rung 5 by 2026-09-30, public 2026-10-01.** If slipping, ship at Rung 4 (train +
  real weights) and name Rung 5 as future work. Never ship a broken higher rung.

## 10. Risks

- **R1 — Attention backward.** Hardest single piece. *Mitigation:* build it as 3 cuBLAS calls +
  custom softmax first, gradient-check each stage, fuse only in Rung 5.
- **R2 — cuBLAS layout confusion.** Row-major C = A·B via column-major cuBLAS is a classic time
  sink. *Mitigation:* one dedicated test with the oracle before anything depends on it.
- **R3 — Rung 4 numerical drift.** FP32 vs HF's FP32 should match to 1e-3; if not, bisect per
  layer with dumped activations.
- **R4 — Losing the plot.** *Mitigation:* this charter; re-read §2/§3 before the writeup.

## 11. After this project

MoE + RoPE (+ SwiGLU / GQA) as an extensions chapter, then the agents + RAG product on open-weight
models. Order lives in `~/Sync/Brain/AI-Context/ml-job-roadmap.md`.

## 12. References

Vaswani 2017 (Attention Is All You Need) · Radford 2019 (GPT-2) · Karpathy, "Let's reproduce GPT-2"
(video, the spec) and llm.c (reference, read only after own piece passes) · Ba 2016 (LayerNorm) ·
Hendrycks 2016 (GELU) · Loshchilov 2019 (AdamW) · Dao 2022 (FlashAttention, for Rung 5) ·
Salmon 2011 (Philox) · Boehm, "How to Optimize a CUDA Matmul Kernel" · PMPP 4th ed.
