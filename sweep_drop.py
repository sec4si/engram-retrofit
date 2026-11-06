"""How much does removing each FFN hurt? Single layers on 64 held-out sequences, then sets on 256."""
import numpy as np
import torch

from exp import batches, load, ppl

held = torch.from_numpy(batches("heldout").astype(np.int64))
for i in range(1, 28):
    _, model, _ = load("base", drop_mlp=(i,))
    print("single", i, round(ppl(model, None, held[:64]), 2), flush=True)
for drop in [(), (3, 4), (3, 4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 7, 8)]:
    _, model, _ = load("base", drop_mlp=drop)
    print("set", list(drop), round(ppl(model, None, held[:256]), 2), flush=True)
