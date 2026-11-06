"""Collapse results.jsonl into one markdown table (latest row per configuration wins)."""
import json

rows = {}
for line in open("results.jsonl"):
    r = json.loads(line)
    rows[(r["mode"], r["ckpt"], r["random_rows"], str(r.get("drop_mlp") or "") + (" +table" if r.get("mean_table") else ""))] = r
print("| run | dropped FFN | random rows | heldout PPL | wiki PPL | TriviaQA EM |")
print("|---|---|---|---|---|---|")
for (mode, ckpt, rnd, drop), r in sorted(rows.items(), key=lambda kv: str(kv[0])):
    print(f"| {ckpt or mode} | {drop} | {'yes' if rnd else ''} | {r['heldout_ppl']:.3f} | {r['wiki_ppl']:.3f} | {r['trivia_em']:.1%} |")
