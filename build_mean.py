"""Average each chosen FFN's output by the trigram, bigram and token ending at each position.

  python build_mean.py 3,4,5,6     # 0-based decoder layers; writes data/meantab.pt
"""
import sys

import numpy as np
import torch

from engram import NGramTable
from exp import batches, forward, load

LAYERS = [int(x) for x in sys.argv[1].split(",")]
TRAIN_SEQS, BS = 19_531, 16

tok, model, _ = load("base")
dictionary = torch.load("data/dict50.pt")
keys = {n: k.sort().values for n, k in dictionary["keys"].items()}
V, D = dictionary["V"], model.config.hidden_size
size = {1: V, 2: len(keys[2]), 3: len(keys[3])}
empty = {l: {n: torch.zeros(0) for n in (1, 2, 3)} for l in LAYERS}
grouper = NGramTable({"V": V, "keys": keys, "means": empty, "seen": empty}, tok, model).cuda()
sums = {l: {n: torch.zeros(size[n], D, device="cuda") for n in (1, 2, 3)} for l in LAYERS}
counts = {l: {n: torch.zeros(size[n], device="cuda") for n in (1, 2, 3)} for l in LAYERS}


def accumulate(l):
    def hook(mod, args, out):
        o = out.float().flatten(0, 1)
        g = grouper._groups
        for n in (1, 2, 3):
            idx, hit = (g[1], torch.ones_like(g[1], dtype=torch.bool)) if n == 1 else g[n]
            idx, hit = idx.flatten(), hit.flatten()
            sums[l][n].index_add_(0, idx[hit], o[hit])
            counts[l][n].index_add_(0, idx[hit], torch.ones(int(hit.sum()), device="cuda"))
    return hook


with torch.no_grad():
    for l in LAYERS:
        model.model.layers[l].mlp.register_forward_hook(accumulate(l))
    train = batches("train")[:TRAIN_SEQS]
    for i in range(0, len(train), BS):
        ids = torch.from_numpy(train[i:i + BS].astype(np.int64)).cuda()
        grouper.set_ids(ids)
        forward(model, None, ids)

means, seen = {}, {}
for l in LAYERS:
    overall = sums[l][1].sum(0) / counts[l][1].sum()
    means[l], seen[l] = {}, {}
    for n in (1, 2, 3):
        m = sums[l][n] / counts[l][n].clamp_min(1)[:, None]
        if n == 1:
            m[counts[l][1] == 0] = overall           # tokens never seen fall back to the overall mean
        means[l][n] = m.to(torch.bfloat16).cpu()
        if n > 1:
            seen[l][n] = (counts[l][n] > 0).cpu()
    print("layer", l, "rows seen:", {n: int((counts[l][n] > 0).sum()) for n in (1, 2, 3)}, flush=True)
torch.save({"V": V, "keys": keys, "means": means, "seen": seen}, "data/meantab.pt")
