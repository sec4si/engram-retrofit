"""Count normalized 2/3-grams over the first N training tokens and keep frequent ones."""
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from engram import compress_vocab

MODEL, N = "models/Qwen3-0.6B-Base", 100_000_000
tok = AutoTokenizer.from_pretrained(MODEL)
cmap, pad = compress_vocab(tok, AutoConfig.from_pretrained(MODEL).vocab_size)
V = pad + 1
c = cmap.numpy()[np.fromfile("data/bin/train.bin", dtype=np.uint32, count=N).astype(np.int64)]
keys = {2: c[:-1] * V + c[1:], 3: (c[:-2] * V + c[1:-1]) * V + c[2:]}
out = {}
for n, k in keys.items():
    uniq, cnt = np.unique(k, return_counts=True)
    for th in (5, 20, 50, 100):
        sel = cnt >= th
        print(f"{n}-gram >= {th}: {sel.sum():,} rows, covers {cnt[sel].sum() / len(k):.1%} of positions")
    out[n] = torch.from_numpy(uniq[cnt >= int(sys.argv[1])]) if len(sys.argv) > 1 else None
if len(sys.argv) > 1:
    torch.save({"V": V, "keys": out}, f"data/dict{sys.argv[1]}.pt")
