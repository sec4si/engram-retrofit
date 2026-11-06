# engram-retrofit

Code and raw results for the Infer article **[Can you bolt a memory table onto a model that was trained without one?](https://infer.sec4.si/research/engram/)**

[Engram](https://arxiv.org/abs/2601.07372) gives a language model large n-gram lookup tables, trained together with the model. We ask whether such a table can instead be added to a model that is already trained and frozen. We freeze Qwen3-0.6B-Base, insert tables before decoder layers 1 and 9, train only the tables and their small projections on FineWeb-Edu, and compare against LoRA trained on the same data.

1. **Hashed tables**, following DeepSeek V4.1 Flash’s released inference code (normalised token ids, multiplicative-XOR hashing into prime-sized buckets, signed-sqrt gate). Held-out perplexity improves at first, then drifts upward; the largest table ends worse than no table.
2. **Codebooks**: one row per n-gram seen at least 20 or 50 times in the training text, found by exact match; every other n-gram is a miss and adds nothing. Training stays flat, held-out perplexity drops 2.4%, and we detect no change on TriviaQA.
3. **Replacing removed layers**: remove some early feed-forward layers (0-based decoder layers 3–6, or 3–4), then repair the damage with a codebook placed just before them or with LoRA.
4. **Counted tables**: average each early FFN’s output by the n-gram ending at each position, measure how much of the output that explains, replace the FFNs with those averages, and repair with LoRA on top.
5. **Random-row controls**: the same lookups with every row replaced by a random other row of the same trained table, to separate what the rows hold from what the projections learn.

Every chart in the article is drawn from `results/data.json`, which is built from the files next to it.

## Setup

Python 3.12, PyTorch with CUDA, and one GPU with about 60 GB of memory for the largest hashed table (the codebooks need about 36 GB). The runs used two RTX PRO 6000 Blackwell cards, one run per card.

```bash
pip install torch transformers datasets peft pyarrow huggingface_hub
```

## Running

```bash
./run_all.sh
```

It downloads the model and data, tokenises them, counts the n-gram dictionaries, and then trains and evaluates every run in the article in turn.

| Step | Script | Output |
|---|---|---|
| Tokenise 400M training tokens, 2M held-out tokens, WikiText-103 test | `exp.py prep` | `data/bin/` |
| Count 2- and 3-grams over the first 100M tokens; keep the frequent ones | `count.py 20`, `count.py 50` | `data/dict20.pt`, `data/dict50.pt` |
| Train (10,000 steps × 32 × 1,024 tokens) | `exp.py train` | `runs/<name>/train.log` |
| Evaluate: held-out and WikiText perplexity, TriviaQA closed-book EM | `exp.py eval` | one line in `results.jsonl` |
| FFN removal sweep: one layer at a time on 64 sequences, then sets on 256 | `sweep_drop.py` | stdout |
| Remove FFNs and repair (`--drop-mlp`, `--engram-layers`) | `exp.py train/eval` | `runs/p2_*` |
| R² of FFN outputs against n-gram averages; FFN replacement test | `rsq.py` | `rsq.json` |
| Counted n-gram tables for chosen layers | `build_mean.py` | `data/meantab.pt` |
| Use counted tables in place of FFNs (`--mean-table`) | `exp.py train/eval` | `runs/p3_*` |
| Summary table | `summarize.py` | stdout |

The Engram modules are in `engram.py`: `EngramLayer` (hashed) and `DictEngramLayer` (codebook) share the gate in `GatedInject`. `smoke.py` and `smoke_dict.py` check that a freshly initialised table leaves the model’s output bit-identical and that gradients reach the rows.

## Results

| Run | Trainable | Held-out PPL | WikiText PPL | TriviaQA EM |
|---|---|---|---|---|
| No table | 0 | 16.04 | 14.11 | 21.25% |
| LoRA r=8 | 5.0M | 15.35 | 14.16 | 19.35% |
| Hashed, 1M rows per head | 2.05B | 16.73 | 15.45 | 19.30% |
| … random rows | | 16.92 | 14.97 | 17.80% |
| Hashed, 200k rows per head | 414M | 16.05 | 14.63 | 19.95% |
| … random rows | | 16.38 | 14.44 | 20.35% |
| Codebook, ≥ 50 | 190M | 15.65 | 14.20 | 21.50% |
| … random rows | | 16.11 | 14.19 | 20.85% |
| Codebook, ≥ 20 | 482M | 15.71 | 14.26 | 21.45% |
| Codebook, ≥ 20, 5× row learning rate | 482M | 15.84 | 14.37 | 20.70% |

With early feed-forward layers removed (5,000 repair steps; the intact model scores 16.04 / 14.11 / 21.25%):

| Layers removed (0-based) | Repair | Trainable | Held-out PPL | WikiText PPL | TriviaQA EM |
|---|---|---|---|---|---|
| 3–6 | none | 0 | 83.17 | 84.40 | 0.35% |
| 3–6 | codebook ≥ 50 | 190M | 20.24 | 19.91 | 10.85% |
| 3–6 | … random rows | | 30.28 | 29.91 | 5.45% |
| 3–6 | LoRA r=8 | 4.7M | 18.62 | 17.99 | 10.45% |
| 3–4 | none | 0 | 35.63 | 34.01 | 7.40% |
| 3–4 | codebook ≥ 50 | 190M | 18.29 | 17.25 | 13.30% |
| 3–4 | … random rows | | 22.71 | 21.21 | 9.35% |
| 3–4 | LoRA r=8 | 4.8M | 17.32 | 16.36 | 10.55% |

Counted tables, from 20M tokens (R² on 524,288 held-out tokens; trigram → bigram → token → overall mean backoff):

| Layer (0-based) | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 13 | 20 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| R² | 0.72 | 0.00 | 0.63 | 0.63 | 0.54 | 0.44 | 0.32 | 0.21 | 0.41 | 0.21 |

With layers 3–6 (0-based) replaced:

| Stand-in | Trainable | Held-out PPL | WikiText PPL | TriviaQA EM |
|---|---|---|---|---|
| counted table, no training | 0 | 20.37 | 20.74 | 12.45% |
| counted table + LoRA r=8 | 4.7M | 17.38 | 16.83 | 13.45% |

`results/rsq.json` also holds perplexity on 256 held-out sequences with FFNs replaced by zeros, one fixed vector, a per-token average or the full lookup.

`results/results.jsonl` holds every evaluation as it was written, including duplicates and a 3,000-step run (`engram_small`) that the article does not use. `results/logs/` holds the training logs.

## Notes and caveats

- **Scope.** One model, one seed: 327.68M training tokens per run in the first two parts (10,000 steps), 163.84M in the repair runs (5,000 steps).
- **TriviaQA noise.** 2,000 questions give a standard error of about 0.9 points; we did not run paired tests.
- **The two hashed tables differ in size and in row learning rate** (5× for the 1M table, 1× for the 200k one).
- **The training recipe is ours.** DeepSeek has released inference code for the tables, not training code.
- **A discarded first attempt.** Our first runs used the chat-tuned `Qwen/Qwen3-0.6B` by mistake; continued training mostly undid its chat tuning, so those runs are not included.

## License

MIT. If you use these results, please cite the article:

> sec4si. “Can you bolt a memory table onto a model that was trained without one?” Infer, 2026. https://infer.sec4.si/research/engram/
