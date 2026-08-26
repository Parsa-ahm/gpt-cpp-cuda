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

## Rung 2 — Reverse-mode autodiff (then its GPU kernels)

- **Andrej Karpathy, "The spelled-out intro to neural networks and backprop: building micrograd"**
  (YouTube). Builds reverse-mode autodiff from zero. It's Python, but the *concept* — the graph, the
  chain rule as a backward pass — transfers directly to what you'll do in C++/CUDA. Watch before you
  design your tape.
- **Baydin et al., "Automatic Differentiation in Machine Learning: a Survey"** — the reference for
  forward vs reverse mode and why reverse is right for scalar-loss training.
- For backward kernels of matmul/conv: derive them yourself from the forward pass; the matmul
  backward is two more matmuls (dA = dC·Bᵀ, dB = Aᵀ·dC) — good to prove on paper first.

## Rungs 3–5 — Score-based generative modeling + sampling + inpainting

- **Yang Song, "Generative Modeling by Estimating Gradients of the Data Distribution"**
  (yang-song.net/blog/2021/score) — the canonical, readable explainer of score matching + annealed
  Langevin, by the person who invented the method. **Read this first for the ML spine.**
- **Lilian Weng, "What are Diffusion Models?"** (lilianweng.github.io) — clear, well-diagrammed
  overview tying score-based ↔ DDPM.
- **Papers:** Vincent 2011 (Denoising Score Matching — your exact training target); Song & Ermon 2019
  (NCSN — the model + annealed Langevin you're building); Ho et al. 2020 (DDPM — the connection).
- **YouTube:** Outlier, *"Diffusion Models | Paper Explanation | Math Explained"* (best visual
  intro); Jia-Bin Huang, *"How I Understand Diffusion Models"* (very clear intuition).
- Murphy Advanced Topics — diffusion / score-based chapter for the rigorous version.

## Situating (for the writeup + interviews)

- Flow matching / rectified flow: Lipman et al. 2022; Liu et al. 2022. (Why frontier labs moved on.)
- Consistency models: Song et al. 2023. (One-step sampling.)

## Profiling tools (Rung 6)

- **Nsight Compute** (`ncu`) and **Nsight Systems** (`nsys`) — installed with the CUDA toolkit. `ncu`
  gives you occupancy, memory throughput, and the roofline for each kernel.
