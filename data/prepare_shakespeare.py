# /// script
# requires-python = ">=3.10"
# dependencies = ["tiktoken>=0.7", "requests>=2.31"]
# ///
"""
TinyShakespeare -> GPT-2 BPE token ids -> flat binary the C++ engine mmaps.

    uv run data/prepare_shakespeare.py

Writes data/train.bin and data/val.bin. Both are gitignored; this script is the
reproducible path from a clean clone.

File format (little-endian, the only format the engine reads):

    offset  bytes  type   meaning
    0       4      int32  magic 0x47505431 ("GPT1")
    4       4      int32  version, currently 1
    8       4      int32  n_tokens, how many ids follow
    12      4      int32  vocab_size the ids were produced against
    16      4*n    int32  the token ids

Why a header at all: the engine can assert the magic, the version, and that
vocab_size matches what it was compiled for. Pointing the trainer at the wrong
.bin then watching the loss curve refuse to fall is a bad afternoon, and four
ints prevent it.

Why int32 and not the uint16 that nanoGPT uses: the engine's embedding kernel
takes `const int*`, so int32 is a straight memcpy to the device with no widening
pass. GPT-2's 50257 ids do fit in uint16 and at production scale you would want
that halving, but this corpus is ~338k tokens, so the file is 1.4 MB either way
and the simpler loader is worth more than the bytes.
"""

import struct
import sys
from pathlib import Path

import requests
import tiktoken

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MAGIC = 0x47505431
VERSION = 1
VAL_FRACTION = 0.1

HERE = Path(__file__).resolve().parent


def write_bin(path: Path, ids: list[int], vocab_size: int) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<iiii", MAGIC, VERSION, len(ids), vocab_size))
        f.write(struct.pack(f"<{len(ids)}i", *ids))


def write_vocab(path: Path, enc: "tiktoken.Encoding") -> None:
    """
    Dump the GPT-2 vocabulary so the C++ sampler can print words instead of ids.

    Format: int32 magic, int32 version, int32 n_tokens, then per token an int32
    byte length followed by that many raw bytes. Decoding in the engine is then
    concatenation and nothing else: no merge table, no regex, no BPE logic.
    Encoding still lives in Python, which is all Rung 3 and Rung 4 need, since
    prompts are short and tokenized ahead of time.
    """
    n = enc.n_vocab
    with path.open("wb") as f:
        f.write(struct.pack("<iii", MAGIC, VERSION, n))
        for i in range(n):
            try:
                raw = enc.decode_single_token_bytes(i)
            except KeyError:
                raw = b""          # a few ids in the 50257 range are unused
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)


def main() -> int:
    raw = HERE / "input.txt"
    if raw.exists():
        text = raw.read_text(encoding="utf-8")
        print(f"using cached {raw.name} ({len(text):,} chars)")
    else:
        print(f"downloading {URL}")
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        raw.write_text(text, encoding="utf-8")
        print(f"wrote {raw.name} ({len(text):,} chars)")

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    vocab_size = enc.n_vocab
    print(f"tokenized: {len(ids):,} tokens, vocab {vocab_size}")

    if max(ids) >= vocab_size:
        print(f"FATAL: token id {max(ids)} >= vocab {vocab_size}", file=sys.stderr)
        return 1

    # Contiguous split, not shuffled. The corpus is one continuous text and the
    # model is trained on contiguous windows, so a shuffled split would leak
    # train context into val windows through the overlap.
    split = int(len(ids) * (1.0 - VAL_FRACTION))
    train_ids, val_ids = ids[:split], ids[split:]

    write_bin(HERE / "train.bin", train_ids, vocab_size)
    write_bin(HERE / "val.bin", val_ids, vocab_size)
    write_vocab(HERE / "vocab.bin", enc)

    print(f"train.bin  {len(train_ids):,} tokens  {16 + 4 * len(train_ids):,} bytes")
    print(f"val.bin    {len(val_ids):,} tokens  {16 + 4 * len(val_ids):,} bytes")
    print(f"vocab.bin  {vocab_size:,} tokens  {(HERE / 'vocab.bin').stat().st_size:,} bytes")

    # Round-trip a sample so a silent encoder change cannot go unnoticed.
    probe = enc.decode(train_ids[:24])
    print(f"first 24 tokens decode to: {probe!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
