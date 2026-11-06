"""Engram-style n-gram memory retrofitted onto a frozen HF decoder.

Follows DeepSeek V4.1's inference/engram.py: normalized token ids, multiplicative-XOR
hashing into per-(order, head) prime buckets, signed-sqrt context gate. The value
projection starts at zero so the wrapped model is unchanged before training.
"""
import math
import re
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_prime(n):
    if n < 2:
        return False
    for p in range(2, math.isqrt(n) + 1):
        if n % p == 0:
            return False
    return True


def primes_from(start, count):
    out, n = [], start
    while len(out) < count:
        n += 1
        if is_prime(n):
            out.append(n)
    return out


def normalize(text):
    """V4.1's normalizer: NFKC, NFD, strip accents, lowercase, collapse whitespace."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return s if s == " " else s.strip()


def compress_vocab(tokenizer, vocab_size):
    """Map every token id to a normalized-surface id; returns (map, pad_id)."""
    keys, cmap = {}, []
    for i in range(vocab_size):
        text = tokenizer.decode([i]) if i < len(tokenizer) else ""
        cmap.append(keys.setdefault(normalize(text) or text, len(keys)))
    pad = len(keys)
    return torch.tensor(cmap, dtype=torch.long), pad


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class GatedInject(nn.Module):
    """h + gate(h, k) * v, with k, v projected from the looked-up memory e."""

    def __init__(self, width, dim):
        super().__init__()
        self.dim = dim
        self.w_k = nn.Linear(width, dim, bias=False)
        self.w_v = nn.Linear(width, dim, bias=False)
        self.norm_h, self.norm_k = RMSNorm(dim), RMSNorm(dim)

    def inject(self, h, e):
        e = e.to(h.dtype)
        key, value = self.w_k(e), self.w_v(e)
        dot = (self.norm_h(h) * self.norm_k(key)).sum(-1, keepdim=True) / math.sqrt(self.dim)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + (gate * value).to(h.dtype)


class EngramLayer(GatedInject):
    def __init__(self, layer_id, dim, orders, heads, head_dim, prime_start, compressed_vocab):
        super().__init__(len(orders) * heads * head_dim, dim)
        self.orders = orders
        gen = torch.Generator().manual_seed(10007 * layer_id)
        bound = max(1, (2**63 - 1) // compressed_vocab // 2)
        mult = torch.randint(0, bound, (max(orders),), generator=gen) * 2 + 1
        primes = torch.tensor(primes_from(prime_start, len(orders) * heads)).view(len(orders), heads)
        offsets = torch.cumsum(primes.flatten(), 0).roll(1).view_as(primes)
        offsets.view(-1)[0] = 0
        self.register_buffer("mult", mult, persistent=False)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.rows = int(primes.sum())
        # Zero rows: the model starts unchanged and buckets never trained stay silent.
        self.embed = nn.Embedding(self.rows, head_dim)
        nn.init.zeros_(self.embed.weight)

    def hash(self, c, pad, random=False):
        """c: [B, T] compressed ids -> [B, T, orders*heads] table rows."""
        rolling, hashes = torch.zeros_like(c), []
        for j in range(max(self.orders)):
            shifted = F.pad(c, (j, 0), value=pad)[:, : c.shape[1]]
            rolling = torch.bitwise_xor(rolling, shifted * self.mult[j])
            if j + 1 in self.orders:
                hashes.append(rolling)
        h = torch.stack(hashes, -1)[..., None]
        rows = (h % self.primes + self.offsets).flatten(-2)
        return torch.randint_like(rows, self.rows) if random else rows

    def forward(self, h, rows):
        return self.inject(h, self.embed(rows).flatten(-2))


class DictEngramLayer(GatedInject):
    """One row per frequent n-gram, found by exact match; anything else is a miss and adds nothing."""

    def __init__(self, dim, keys, vocab, order_dim):
        super().__init__(len(keys) * order_dim, dim)
        self.orders, self.vocab = sorted(keys), vocab
        for n in self.orders:
            self.register_buffer(f"keys{n}", keys[n].sort().values, persistent=False)
        self.tables = nn.ModuleList(nn.Embedding(len(keys[n]), order_dim) for n in self.orders)
        for t in self.tables:
            nn.init.zeros_(t.weight)

    def hash(self, c, pad, random=False):
        rows = []
        for n in self.orders:
            keys, q = getattr(self, f"keys{n}"), torch.zeros_like(c)
            for j in reversed(range(n)):
                q = q * self.vocab + F.pad(c, (j, 0), value=pad)[:, : c.shape[1]]
            idx = torch.searchsorted(keys, q).clamp_max(len(keys) - 1)
            hit = keys[idx] == q
            rows.append((torch.randint_like(idx, len(keys)) if random else idx, hit))
        return rows

    def forward(self, h, rows):
        e = torch.cat([t(idx) * hit[..., None] for t, (idx, hit) in zip(self.tables, rows)], -1)
        return self.inject(h, e)


class Engram(nn.Module):
    def __init__(self, tokenizer, model, layer_ids, orders=(2, 3), heads=4, head_dim=128,
                 rows_per_head=1_000_000, dictionary=None, order_dim=256):
        super().__init__()
        cmap, self.pad = compress_vocab(tokenizer, model.config.vocab_size)
        self.register_buffer("cmap", cmap, persistent=False)
        start = rows_per_head
        self.layers = nn.ModuleDict()
        for lid in layer_ids:
            if dictionary:
                self.layers[str(lid)] = DictEngramLayer(model.config.hidden_size, dictionary["keys"],
                                                        dictionary["V"], order_dim)
                continue
            layer = EngramLayer(lid, model.config.hidden_size, orders, heads, head_dim,
                                start, self.pad + 1)
            start = int(layer.primes.max())
            self.layers[str(lid)] = layer
        self.enabled = True
        self.random_rows = False  # ablation: same lookups, wrong rows
        self._rows = {}

    def set_ids(self, input_ids):
        c = self.cmap[input_ids]
        self._rows = {k: m.hash(c, self.pad, self.random_rows) for k, m in self.layers.items()}

    def attach(self, model):
        for lid, layer in self.layers.items():
            def hook(mod, args, kwargs, layer=layer, lid=lid):
                if not self.enabled:
                    return None
                if args:
                    return (layer(args[0], self._rows[lid]), *args[1:]), kwargs
                kwargs["hidden_states"] = layer(kwargs["hidden_states"], self._rows[lid])
                return args, kwargs
            model.model.layers[int(lid)].register_forward_pre_hook(hook, with_kwargs=True)
        return self


class NGramTable(nn.Module):
    """Frozen stand-in for whole FFNs: the mean FFN output of the trigram, bigram or token ending at
    each position, backing off in that order. Built by build_mean.py from the intact model."""

    def __init__(self, tables, tokenizer, model):
        super().__init__()
        cmap, self.pad = compress_vocab(tokenizer, model.config.vocab_size)
        assert tables["V"] == self.pad + 1
        self.register_buffer("cmap", cmap, persistent=False)
        self.vocab, self.layers = tables["V"], sorted(tables["means"])
        for n in (2, 3):
            self.register_buffer(f"keys{n}", tables["keys"][n], persistent=False)
        for l in self.layers:
            for n in (1, 2, 3):
                self.register_buffer(f"mean{l}_{n}", tables["means"][l][n], persistent=False)
            for n in (2, 3):
                self.register_buffer(f"seen{l}_{n}", tables["seen"][l][n], persistent=False)
        self._groups = None

    def set_ids(self, input_ids):
        c = self.cmap[input_ids]
        prev = [F.pad(c, (j, 0), value=self.pad)[:, : c.shape[1]] for j in range(3)]
        self._groups = {1: c}
        for n in (2, 3):
            keys, q = getattr(self, f"keys{n}"), torch.zeros_like(c)
            for j in reversed(range(n)):
                q = q * self.vocab + prev[j]
            idx = torch.searchsorted(keys, q).clamp_max(len(keys) - 1)
            self._groups[n] = (idx, keys[idx] == q)

    def predict(self, l):
        pred = getattr(self, f"mean{l}_1")[self._groups[1]].clone()
        for n in (2, 3):
            idx, hit = self._groups[n]
            hit = hit & getattr(self, f"seen{l}_{n}")[idx]
            pred[hit] = getattr(self, f"mean{l}_{n}")[idx[hit]]
        return pred

    def stand_in(self, l):
        return _StandIn(self, l)


class _StandIn(nn.Module):
    def __init__(self, table, layer):
        super().__init__()
        object.__setattr__(self, "table", table)  # shared, not a child: keeps it out of state dicts
        self.layer = layer

    def forward(self, x):
        return self.table.predict(self.layer).to(x.dtype)
