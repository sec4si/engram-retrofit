import torch
from exp import load, forward, lm_loss
tok, model, _ = load("base")
ids = tok(["The United States of America is a country in North America, and the capital is Washington."], return_tensors="pt")["input_ids"].cuda()
base = forward(model, None, ids)
tok, model, eg = load("engram", dict_path="data/dict20.pt")
print("identical at init:", torch.equal(base, forward(model, eg, ids)))
rows = eg._rows["1"]
print("hit rate per order:", [round(float(h.float().mean()), 3) for _, h in rows])
opt = torch.optim.AdamW(eg.parameters(), lr=1e-3)
for step in range(10):
    loss = lm_loss(forward(model, eg, ids), ids); loss.backward()
    if step in (0, 9):
        g = eg.layers["1"].tables[0].weight.grad
        print(step, "loss", round(loss.item(), 4), "grad rows", int((g.abs().sum(-1) > 0).sum()))
    opt.step(); opt.zero_grad()
eg.random_rows = True; forward(model, eg, ids); print("random rows ok")
