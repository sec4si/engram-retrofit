"""How much of a feed-forward layer's output is a lookup on the n-gram ending at that position?

Pass 1 (20M training tokens): per layer, average the FFN output by trigram, bigram and current token,
using the >= 50 codebook keys for trigrams and bigrams.
Pass 2 (held-out tokens): R^2 of predicting the FFN output from those averages, with backoff
trigram -> bigram -> token, against the global mean.
Pass 3: replace whole FFNs by those predictions and measure perplexity, against zeros and the global mean.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from exp import batches, forward, lm_loss, load

LAYERS = [1, 2, 3, 4, 5, 6, 9, 13, 20, 27]   # 0-based decoder layers
TRAIN_SEQS, HELD_SEQS, PPL_SEQS, BS = 19_531, 500, 256, 16  # the R^2 loop rounds HELD_SEQS up to whole batches: 512 sequences
SETS = [[1], [2], [3], [3, 4], [3, 4, 5, 6]]

tok, model, eg = load("engram", dict_path="data/dict50.pt", engram_layers=[0])
eg.enabled = False                             # only used for its token normalisation
dictionary = torch.load("data/dict50.pt")
V, keys = dictionary["V"], {n: k.cuda() for n, k in dictionary["keys"].items()}
D = model.config.hidden_size


def groups(ids):
    """Per position: compressed token id, and (index, hit) for the bigram and trigram codebooks."""
    c = eg.cmap[ids]
    prev = [F.pad(c, (j, 0), value=eg.pad)[:, : c.shape[1]] for j in range(3)]
    out = {1: (c, torch.ones_like(c, dtype=torch.bool))}
    for n in (2, 3):
        q = torch.zeros_like(c)
        for j in reversed(range(n)):
            q = q * V + prev[j]
        idx = torch.searchsorted(keys[n], q).clamp_max(len(keys[n]) - 1)
        out[n] = (idx, keys[n][idx] == q)
    return out


size = {1: V, 2: len(keys[2]), 3: len(keys[3])}
sums = {l: {n: torch.zeros(size[n], D, device="cuda") for n in (1, 2, 3)} for l in LAYERS}
counts = {l: {n: torch.zeros(size[n], device="cuda") for n in (1, 2, 3)} for l in LAYERS}
state = {}


def accumulate(l):
    def hook(mod, args, out):
        o = out.float().flatten(0, 1)
        for n, (idx, hit) in state["g"].items():
            idx, hit = idx.flatten(), hit.flatten()
            sums[l][n].index_add_(0, idx[hit], o[hit])
            counts[l][n].index_add_(0, idx[hit], torch.ones(int(hit.sum()), device="cuda"))
    return hook


def predict(l, g, scheme):
    """Predicted FFN output for scheme in {global, token, bigram, trigram, codebook}."""
    mean = state["mean"][l]
    shape = g[1][0].shape
    pred = mean.expand(*shape, D).clone()
    order = {"global": [], "token": [1], "bigram": [1, 2], "trigram": [1, 2, 3], "codebook": [2, 3]}[scheme]
    for n in order:
        idx, hit = g[n]
        hit = hit & (counts[l][n][idx] > 0)
        pred[hit] = sums[l][n][idx[hit]]
    return pred


with torch.no_grad():
    # Pass 1: accumulate sums by group.
    handles = [model.model.layers[l].mlp.register_forward_hook(accumulate(l)) for l in LAYERS]
    train = batches("train")[:TRAIN_SEQS]
    for i in range(0, len(train), BS):
        ids = torch.from_numpy(train[i:i + BS].astype(np.int64)).cuda()
        state["g"] = groups(ids)
        forward(model, None, ids)
    for h in handles:
        h.remove()
    state["mean"] = {l: sums[l][1].sum(0) / counts[l][1].sum() for l in LAYERS}
    for l in LAYERS:
        for n in (1, 2, 3):
            sums[l][n] /= counts[l][n].clamp_min(1)[:, None]   # sums now hold group means

    # Pass 2: R^2 on held-out tokens.
    SCHEMES = ["global", "token", "bigram", "trigram", "codebook"]
    ss = {l: {s: 0.0 for s in SCHEMES} for l in LAYERS}
    captured = {}

    def capture(l):
        def hook(mod, args, out):
            captured[l] = out.float()
        return hook

    handles = [model.model.layers[l].mlp.register_forward_hook(capture(l)) for l in LAYERS]
    held = batches("heldout")
    for i in range(0, HELD_SEQS, BS):
        ids = torch.from_numpy(held[i:i + BS].astype(np.int64)).cuda()
        g = groups(ids)
        forward(model, None, ids)
        for l in LAYERS:
            for s in SCHEMES:
                ss[l][s] += float(((captured[l] - predict(l, g, s)) ** 2).sum())
    for h in handles:
        h.remove()
    rsq = {l: {s: 1 - ss[l][s] / ss[l]["global"] for s in SCHEMES[1:]} for l in LAYERS}
    hits = {n: float(groups(torch.from_numpy(held[:HELD_SEQS].astype(np.int64)).cuda())[n][1].float().mean()) for n in (2, 3)}
    print("hit rate on held-out:", hits, flush=True)
    for l in LAYERS:
        print("R2 layer", l, {s: round(v, 4) for s, v in rsq[l].items()}, flush=True)

    # Pass 3: replace FFNs by their predictions.
    def replace(l, scheme):
        def hook(mod, args, out):
            return torch.zeros_like(out) if scheme == "zero" else predict(l, state["g"], scheme).to(out.dtype)
        return hook

    seqs = held[:PPL_SEQS]

    def ppl(hooks):
        handles = [model.model.layers[l].mlp.register_forward_hook(h) for l, h in hooks]
        nll = 0.0
        for i in range(0, len(seqs), BS):
            ids = torch.from_numpy(seqs[i:i + BS].astype(np.int64)).cuda()
            state["g"] = groups(ids)
            nll += lm_loss(forward(model, None, ids), ids).item() * ids.shape[0]
        for h in handles:
            h.remove()
        return float(np.exp(nll / len(seqs)))

    swap = {"intact": ppl([])}
    print("intact", swap["intact"], flush=True)
    for layers in SETS:
        for scheme in ["zero", "global", "token", "trigram"]:
            v = ppl([(l, replace(l, scheme)) for l in layers])
            swap[f"{layers}:{scheme}"] = v
            print("replace", layers, scheme, round(v, 3), flush=True)

json.dump({"layers": LAYERS, "train_tokens": TRAIN_SEQS * 1024, "held_tokens": HELD_SEQS * 1024,
           "hit_rate": hits, "rsq": rsq, "ppl_seqs": PPL_SEQS, "swap": swap},
          open("rsq.json", "w"), indent=1)
