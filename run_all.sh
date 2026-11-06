#!/bin/sh
# Every run in the article, in order, on one GPU. Each train is followed by its evaluations;
# results land in results.jsonl (one JSON line per evaluation) and runs/<name>/train.log.
set -e
PY=${PY:-python}

hf download Qwen/Qwen3-0.6B-Base --local-dir models/Qwen3-0.6B-Base
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
    --include "sample/10BT/001_00000.parquet" "sample/10BT/002_00000.parquet" --local-dir data/fineweb-edu
hf download mandarjoshi/trivia_qa --repo-type dataset --include "rc.nocontext/validation-*" --local-dir data/trivia_qa
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/test-*" --local-dir data/wikitext

$PY exp.py prep                       # 400M training tokens, 2M held-out, WikiText-103 test
$PY count.py 20 && $PY count.py 50    # frequent n-gram dictionaries from the first 100M tokens

engram() {  # engram NAME ARGS...
    name=$1; shift
    $PY exp.py train --mode engram --out runs/$name "$@"
    $PY exp.py eval --mode engram --ckpt runs/$name "$@"
    $PY exp.py eval --mode engram --ckpt runs/$name "$@" --random-rows
}

$PY exp.py eval --mode base
$PY exp.py train --mode lora --rank 8 --lr 2e-4 --out runs/lora8
$PY exp.py eval --mode lora --ckpt runs/lora8

engram engram   --lr 1e-3 --table-lr-mult 5                          # hashed, 1M rows per head
engram hash200k --lr 1e-3 --table-lr-mult 1 --rows-per-head 200000  # hashed, 200k rows per head
engram dict20     --lr 1e-3 --table-lr-mult 1 --dict data/dict20.pt
engram dict50     --lr 1e-3 --table-lr-mult 1 --dict data/dict50.pt
engram dict20_lr5 --lr 1e-3 --table-lr-mult 5 --dict data/dict20.pt

$PY sweep_drop.py

# Chapter 3: remove early FFNs (0-based decoder layers 3-6, or 3-4), then repair with the codebook or LoRA.
for level in heavy:3,4,5,6 mid:3,4; do
    name=${level%%:*} drop=${level#*:}
    $PY exp.py eval --mode base --drop-mlp $drop
    engram p2_${name}_dict --steps 5000 --lr 1e-3 --table-lr-mult 1 --dict data/dict50.pt --engram-layers 3,9 --drop-mlp $drop
    $PY exp.py train --mode lora --rank 8 --lr 2e-4 --steps 5000 --out runs/p2_${name}_lora --drop-mlp $drop
    $PY exp.py eval --mode lora --ckpt runs/p2_${name}_lora --drop-mlp $drop
done


# Chapter 4: how much of each FFN is a lookup, then counted tables in place of layers 3-6, alone and with LoRA.
$PY rsq.py                                   # writes rsq.json
$PY build_mean.py 3,4,5,6                    # writes data/meantab.pt
$PY exp.py eval --mode base --mean-table data/meantab.pt
$PY exp.py train --mode lora --rank 8 --lr 2e-4 --steps 5000 --out runs/p3_mean_lora --mean-table data/meantab.pt
$PY exp.py eval --mode lora --ckpt runs/p3_mean_lora --mean-table data/meantab.pt

$PY summarize.py
