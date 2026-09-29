# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.40", "numpy>=1.26"]
# ///
"""
Rung 4: export HuggingFace GPT-2 124M into the engine's flat parameter format,
plus reference logits for a fixed prompt.

    uv run data/export_gpt2.py

Writes:
    data/gpt2_124M.bin        header + 16 parameter tensors, engine order
    data/gpt2_ref_logits.bin  header + prompt ids + reference logits (T, V)

The Rung 4 check is: load gpt2_124M.bin into the engine, run one forward on the
prompt ids, and compare against gpt2_ref_logits.bin. Max absolute difference
must be under 1e-3 in fp32.

WHY NO TRANSPOSES ANYWHERE BELOW

HuggingFace GPT-2 uses Conv1D, not Linear, for every projection. Conv1D stores
its weight as (in_features, out_features). The engine's launch_linear_fwd calls
sgemm_rm(false, false, N, OC, C, ...), i.e. out(N,OC) = x(N,C) @ W(C,OC), which
wants exactly (in, out) too. So every weight copies across verbatim. Had the
engine been written against torch.nn.Linear's (out, in) convention this script
would need sixteen transposes and Rung 4 would be a much worse afternoon.

The LM head is not exported: GPT-2 ties it to wte, and so does the engine.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

MAGIC_W = 0x47505457  # "GPTW"
MAGIC_R = 0x47505252  # "GPRR"
VERSION = 1
PROMPT = "The capital of France is"

HERE = Path(__file__).resolve().parent


def main() -> int:
    print("loading gpt2 from HuggingFace (downloads ~500 MB on first run)")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()
    cfg = model.config

    L, C, V, NH, maxT = cfg.n_layer, cfg.n_embd, cfg.vocab_size, cfg.n_head, cfg.n_positions
    print(f"config: n_layer={L} n_head={NH} n_embd={C} vocab={V} ctx={maxT}")

    def t(name: str) -> np.ndarray:
        return sd[name].detach().cpu().numpy().astype(np.float32)

    def per_layer(suffix: str) -> np.ndarray:
        # Stack layer 0..L-1 contiguously, which is the (L, ...) layout the
        # engine strides through with `+ l * <size>`.
        return np.concatenate([t(f"transformer.h.{i}.{suffix}").ravel() for i in range(L)])

    # The 16 tensors, in the engine's order. Do not reorder: GPT::build walks
    # this exact sequence when it hands out pointers into the flat buffer.
    tensors = [
        ("wte", t("transformer.wte.weight")),                 # (V, C)
        ("wpe", t("transformer.wpe.weight")),                 # (maxT, C)
        ("ln1w", per_layer("ln_1.weight")),                   # (L, C)
        ("ln1b", per_layer("ln_1.bias")),
        ("qkvw", per_layer("attn.c_attn.weight")),            # (L, C, 3C)
        ("qkvb", per_layer("attn.c_attn.bias")),              # (L, 3C)
        ("attprojw", per_layer("attn.c_proj.weight")),        # (L, C, C)
        ("attprojb", per_layer("attn.c_proj.bias")),
        ("ln2w", per_layer("ln_2.weight")),
        ("ln2b", per_layer("ln_2.bias")),
        ("fcw", per_layer("mlp.c_fc.weight")),                # (L, C, 4C)
        ("fcb", per_layer("mlp.c_fc.bias")),
        ("fcprojw", per_layer("mlp.c_proj.weight")),          # (L, 4C, C)
        ("fcprojb", per_layer("mlp.c_proj.bias")),
        ("lnfw", t("transformer.ln_f.weight")),               # (C)
        ("lnfb", t("transformer.ln_f.bias")),
    ]

    expected = {
        "wte": V * C, "wpe": maxT * C,
        "ln1w": L * C, "ln1b": L * C,
        "qkvw": L * C * 3 * C, "qkvb": L * 3 * C,
        "attprojw": L * C * C, "attprojb": L * C,
        "ln2w": L * C, "ln2b": L * C,
        "fcw": L * C * 4 * C, "fcb": L * 4 * C,
        "fcprojw": L * 4 * C * C, "fcprojb": L * C,
        "lnfw": C, "lnfb": C,
    }
    total = 0
    for name, arr in tensors:
        n = arr.size
        if n != expected[name]:
            print(f"FATAL: {name} has {n} elements, engine expects {expected[name]}", file=sys.stderr)
            return 1
        total += n
        print(f"  {name:<10} {n:>12,}")
    print(f"  {'TOTAL':<10} {total:>12,} parameters, {total * 4 / 1e6:.1f} MB")

    out = HERE / "gpt2_124M.bin"
    with out.open("wb") as f:
        f.write(struct.pack("<iiiiiii", MAGIC_W, VERSION, maxT, V, L, NH, C))
        for _, arr in tensors:
            f.write(arr.ravel().astype("<f4").tobytes())
    print(f"wrote {out.name}  {out.stat().st_size:,} bytes")

    # ---- reference logits for the Rung 4 correctness check
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok.encode(PROMPT)
    print(f"prompt {PROMPT!r} -> {len(ids)} tokens {ids}")

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()  # (T, V)

    ref = HERE / "gpt2_ref_logits.bin"
    with ref.open("wb") as f:
        f.write(struct.pack("<iiii", MAGIC_R, VERSION, len(ids), V))
        f.write(np.asarray(ids, dtype="<i4").tobytes())
        f.write(logits.astype("<f4").tobytes())
    print(f"wrote {ref.name}  {ref.stat().st_size:,} bytes")

    nxt = int(logits[-1].argmax())
    print(f"reference next token after the prompt: {nxt} {tok.decode([nxt])!r}")
    print(f"last-position logits: min {logits[-1].min():.4f} max {logits[-1].max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
