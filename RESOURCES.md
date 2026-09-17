# Resources

Curated, high-signal references organized by rung. Not a dump — these are the ones worth your
time. Concepts first; you write the code.

## The two textbooks to have open

- **Programming Massively Parallel Processors (PMPP)** — Kirk, Hwu, El Hajj (4th ed). THE CUDA
  textbook. Chapters on memory model, tiled matmul, and memory coalescing are the backbone of this
  whole project. Read alongside Rungs 1–3.
- **Probabilistic Machine Learning: Advanced Topics** — Murphy (you have it as `book2.pdf` in
  `pml-notes`). Chapters on EBMs, MCMC, and diffusion are the ML spine. Read alongside Rungs 2–5.

---

## Rung 0 — Philox PRNG

- **Paper (primary):** Salmon, Moraes, Dror, Shaw, *"Parallel Random Numbers: As Easy as 1, 2, 3"*
  (SC'11). Section on Philox is short and concrete — derive the round from here.
- **Box-Muller transform:** Wikipedia "Box–Muller transform" is enough; or *Numerical Recipes* ch.7.
  Watch the `log(0)` / `u ∈ (0,1]` edge case.
- Concept of `mulhi`/`mullo` (the 32×32→64 multiply): CUDA has `__umulhi(a,b)`; on the host you use a
  64-bit multiply and take the high/low words. Understand *why* multiply-and-mix is a good
  bit-diffuser.

## Rung 1 — GPU core kernels + tiled SGEMM (the systems heart)

- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"**
  (siboehm.com/articles/22/CUDA-MMM). THE step-by-step walkthrough from naive → tiled → register
  blocking → vectorized, with % of cuBLAS at each step. This is basically your Rung 1 + Rung 6 map.
  Read it, then build it yourself.
- **NVIDIA CUDA C++ Programming Guide** + **Best Practices Guide** (docs.nvidia.com) — memory
  coalescing, shared memory, occupancy. Reference, not cover-to-cover.
- **GPU MODE** (formerly CUDA MODE) YouTube lecture series — modern, practical kernel engineering
  (matmul, memory, profiling). Excellent once you're past the basics.
- PMPP chapters 3–6.

## Rung 2 — Transformer ops, forward + backward

- **Karpathy, "Let's reproduce GPT-2 (124M)"** (YouTube, 4h). THE spec. Watch before coding; do not
  code along in PyTorch. Note every shape.
- **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** — the smaller precursor;
  attention explained slowest here.
- **Backward passes by hand:** derive each on paper before writing the kernel. LayerNorm backward
  and softmax-cross-entropy backward are the two people get wrong. Murphy PML1 ch.13 (backprop)
  and the CS231n notes on backprop are enough.
- **cuBLAS row-major trick:** cuBLAS is column-major; a row-major C = A·B is the column-major
  C^T = B^T·A^T, so call `cublasSgemm` with the operands swapped. One test, then never think
  about it again.
- **llm.c** (Karpathy) — the reference implementation of exactly this project in C/CUDA. Rule:
  read a piece of it only *after* your version of that piece passes its test.

## Rung 3 — Training

- GPT-2 paper (Radford 2019) §2 for the architecture table. AdamW (Loshchilov 2019).
  Cosine schedule with warmup as in the Karpathy video. TinyShakespeare from Karpathy's
  char-rnn repo; `tiktoken` GPT-2 BPE if using real tokens.

## Rung 4 — Real weights

- HuggingFace `transformers` GPT2LMHeadModel for the export script and reference logits (Python,
  outside the engine). Watch for the Conv1D weight layout (HF stores W as [in, out]).

## Rung 5 — Fused attention + benchmarks

- **Dao et al. 2022, FlashAttention** — the algorithm (online softmax, tiling over K/V). Read §3.
- **GPU MODE lectures** (YouTube) on attention kernels and profiling with Nsight Compute.
- Boehm's SGEMM article again for the chart format: % of cuBLAS per kernel version.
- PMPP ch. 5-6 (memory, tiling), ch. 10 (reduction) for the softmax kernel.
