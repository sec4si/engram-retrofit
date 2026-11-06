import torch
from exp import load, forward, lm_loss
tok, model, _ = load("base")
ids = tok(["The capital of France is Paris, and the Eiffel Tower stands there."], return_tensors="pt")["input_ids"].cuda()
base = forward(model, None, ids)
tok, model, eg = load("engram")
print("rows per layer", {k: m.rows for k, m in eg.layers.items()}, "compressed vocab", eg.pad)
out = forward(model, eg, ids)
print("identical at init:", torch.equal(base, out))
r = eg._rows["1"]; print("row range ok:", int(r.min()) >= 0 and int(r.max()) < eg.layers["1"].rows)
opt = torch.optim.AdamW(eg.parameters(), lr=1e-3)
for step in range(20):
    loss = lm_loss(forward(model, eg, ids), ids); loss.backward()
    g = eg.layers["1"].embed.weight.grad
    if step in (0, 1, 19):
        print(step, "loss", round(loss.item(), 4), "table grad nnz rows", int((g.abs().sum(-1) > 0).sum()),
              "w_v grad", float(eg.layers["1"].w_v.weight.grad.abs().sum()))
    opt.step(); opt.zero_grad()
print("mem GB", torch.cuda.max_memory_allocated() / 1e9)
