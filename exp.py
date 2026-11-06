"""Engram retrofit experiment on a frozen Qwen3-0.6B.

  python exp.py prep
  python exp.py train --mode engram|lora --out runs/NAME
  python exp.py eval  --mode base|engram|lora [--ckpt runs/NAME]
"""
import argparse
import glob
import json
import math
import re
import string
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram import Engram, NGramTable

MODEL = "models/Qwen3-0.6B-Base"
SEQ = 1024
ENGRAM_LAYERS = (1, 9)


class NoMLP(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def load(mode, ckpt=None, rank=8, rows_per_head=1_000_000, dict_path=None, drop_mlp=(),
         engram_layers=ENGRAM_LAYERS, mean_table=None):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    model.requires_grad_(False)
    for i in drop_mlp:
        model.model.layers[i].mlp = NoMLP()
    engram = None
    if mean_table:  # frozen n-gram stand-ins for whole FFNs; forward() feeds it the ids like an Engram
        engram = NGramTable(torch.load(mean_table), tok, model).cuda()
        for l in engram.layers:
            model.model.layers[l].mlp = engram.stand_in(l)
    if mode == "engram":
        dictionary = torch.load(dict_path) if dict_path else None
        engram = Engram(tok, model, engram_layers, rows_per_head=rows_per_head,
                        dictionary=dictionary).cuda().attach(model)
        if ckpt:
            engram.load_state_dict(torch.load(f"{ckpt}/engram.pt"))
    if mode == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model
        if ckpt:
            model = PeftModel.from_pretrained(model, ckpt)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, cfg)
    return tok, model, engram


def forward(model, engram, ids, **kw):
    if engram is not None:
        engram.set_ids(ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=ids, use_cache=False, **kw).logits


def lm_loss(logits, ids):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())


# ---------------------------------------------------------------- data

def pack(tok, texts, limit):
    out, n = [], 0
    for i in range(0, len(texts), 2048):
        for ids in tok(texts[i:i + 2048], add_special_tokens=False)["input_ids"]:
            out.append(np.asarray(ids + [tok.eos_token_id], dtype=np.uint32))
            n += len(ids) + 1
        if n >= limit:
            break
    flat = np.concatenate(out)
    return flat[: len(flat) // SEQ * SEQ].reshape(-1, SEQ)


def prep(args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    Path("data/bin").mkdir(exist_ok=True)
    shards = sorted(glob.glob("data/fineweb-edu/sample/10BT/*.parquet"))
    train = [t for s in shards[:-1] for t in pq.read_table(s, columns=["text"])["text"].to_pylist()]
    held = pq.read_table(shards[-1], columns=["text"])["text"].to_pylist()
    wiki = pq.read_table(glob.glob("data/wikitext/wikitext-103-raw-v1/test-*")[0])["text"].to_pylist()
    for name, texts, limit in [("train", train, args.train_tokens), ("heldout", held, 2_000_000),
                               ("wiki", ["".join(wiki)], 10**9)]:
        arr = pack(tok, texts, limit)
        arr.tofile(f"data/bin/{name}.bin")
        print(name, arr.shape, flush=True)


def batches(name):
    return np.fromfile(f"data/bin/{name}.bin", dtype=np.uint32).reshape(-1, SEQ)


# ---------------------------------------------------------------- train

def train(args):
    tok, model, engram = load(args.mode, rank=args.rank, rows_per_head=args.rows_per_head,
                              dict_path=args.dict, drop_mlp=args.drop_mlp,
                              engram_layers=args.engram_layers, mean_table=args.mean_table)
    if args.mode == "engram":
        tables = [p for n, p in engram.named_parameters() if "embed" in n or "tables" in n]
        rest = [p for n, p in engram.named_parameters() if not ("embed" in n or "tables" in n)]
        groups = [{"params": tables, "lr": args.lr * args.table_lr_mult}, {"params": rest, "lr": args.lr}]
        trainable = engram
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        trainable = model
    n_params = sum(p.numel() for g in groups for p in g["params"])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    base_lrs = [g["lr"] for g in opt.param_groups]
    data, held = batches("train"), torch.from_numpy(batches("heldout")[:256].astype(np.int64))
    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")
    print(json.dumps({**vars(args), "trainable_params": n_params}), file=log, flush=True)

    t0, seen = time.time(), 0
    for step in range(1, (args.stop or args.steps) + 1):
        scale = min(step / 200, 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        total = 0.0
        for _ in range(args.accum):
            idx = rng.integers(0, len(data), args.micro)
            ids = torch.from_numpy(data[idx].astype(np.int64)).cuda()
            loss = lm_loss(forward(model, engram, ids), ids) / args.accum
            loss.backward()
            total += loss.item()
            seen += ids.numel()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            rec = {"step": step, "loss": round(total, 4), "tok_s": int(seen / (time.time() - t0))}
            if step % 1000 == 0 or step == args.steps or step == args.stop:
                rec["held_ppl"] = round(ppl(model, engram, held), 3)
            print(json.dumps(rec), file=log, flush=True)

    if args.mode == "engram":
        torch.save(engram.state_dict(), out / "engram.pt")
    else:
        model.save_pretrained(out)


# ---------------------------------------------------------------- eval

@torch.no_grad()
def ppl(model, engram, seqs, bs=16):
    nll, n = 0.0, 0
    for i in range(0, len(seqs), bs):
        ids = seqs[i:i + bs].cuda()
        nll += lm_loss(forward(model, engram, ids), ids).item() * ids.shape[0]
        n += ids.shape[0]
    return math.exp(nll / n)


def norm_answer(s):
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@torch.no_grad()
def trivia(tok, model, engram, n=2000, shots=5, bs=64):
    rows = pq.read_table(sorted(glob.glob("data/trivia_qa/rc.nocontext/validation-*"))).to_pylist()
    prefix = "".join(f"Question: {r['question']}\nAnswer: {r['answer']['value']}\n\n" for r in rows[:shots])
    items = rows[shots:shots + n]
    tok.padding_side = "left"
    hits = 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = tok([f"{prefix}Question: {r['question']}\nAnswer:" for r in chunk],
                  return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
        start = ids.shape[1]
        for _ in range(16):
            pos = (mask.cumsum(-1) - 1).clamp_min(0)
            nxt = forward(model, engram, ids, attention_mask=mask, position_ids=pos)[:, -1].argmax(-1)
            ids = torch.cat([ids, nxt[:, None]], 1)
            mask = torch.cat([mask, torch.ones_like(nxt)[:, None]], 1)
        for r, gen in zip(chunk, tok.batch_decode(ids[:, start:])):
            pred = norm_answer(re.split(r"\n|<\|", gen)[0])
            hits += pred in {norm_answer(a) for a in r["answer"]["normalized_aliases"]}
    return hits / len(items)


def evaluate(args):
    tok, model, engram = load(args.mode, args.ckpt, rows_per_head=args.rows_per_head,
                              dict_path=args.dict, drop_mlp=args.drop_mlp,
                              engram_layers=args.engram_layers, mean_table=args.mean_table)
    model.eval()
    if args.random_rows:
        engram.random_rows = True
    held = torch.from_numpy(batches("heldout").astype(np.int64))
    wiki = torch.from_numpy(batches("wiki").astype(np.int64))
    res = {"mode": args.mode, "ckpt": args.ckpt, "dict": args.dict, "random_rows": args.random_rows,
           "drop_mlp": args.drop_mlp, "mean_table": args.mean_table,
           "heldout_ppl": ppl(model, engram, held), "wiki_ppl": ppl(model, engram, wiki),
           "trivia_em": trivia(tok, model, engram)}
    print(json.dumps(res))
    with open("results.jsonl", "a") as f:
        print(json.dumps(res), file=f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--mode", default="engram", choices=["base", "engram", "lora"])
    ap.add_argument("--out", default="runs/engram")
    ap.add_argument("--ckpt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--rows-per-head", type=int, default=1_000_000)
    ap.add_argument("--table-lr-mult", type=float, default=5.0)
    ap.add_argument("--stop", type=int, help="end early but keep the --steps schedule")
    ap.add_argument("--engram-layers", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(ENGRAM_LAYERS))
    ap.add_argument("--drop-mlp", type=lambda s: [int(x) for x in s.split(",")], default=[],
                    help="comma-separated layers whose FFN is removed")
    ap.add_argument("--mean-table", help="n-gram FFN stand-ins from build_mean.py")
    ap.add_argument("--dict", help="frequent n-gram dictionary from count.py (replaces hashing)")
    ap.add_argument("--random-rows", action="store_true")
    ap.add_argument("--train-tokens", type=int, default=400_000_000)
    args = ap.parse_args()
    {"prep": prep, "train": train, "eval": evaluate}[args.cmd](args)
