"""Stable-ID local layer: delete/add neurons without array compaction.

The old implementation used np.delete, shifting indices and forcing a
downstream rebuild.  Here every neuron keeps a stable slot; deletion only
sets alive=False; addition appends new slots.  Downstream indices never move.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H, W = 96, 96
K = 20
FREQ = 10
CAP0 = 5000
CAPMAX = 6000
TOPK = 1000


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)[:K]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, NLAYER[0]), dtype=np.float32))
    pick = np.sort(rng.choice(NLAYER[-1], CAP0, replace=False))

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        n_tr = 20 if k < FREQ else 5
        for j in range(n_tr + 5):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < n_tr)
    labels = np.array(labels); train = np.array(train)
    X = np.zeros((len(images), len(pick)), dtype=np.float32)
    for i, img in enumerate(images):
        X[i] = model.forward_numpy(encode(img[None]))[-1][0][pick] > 0

    # stable slots: capacity CAPMAX, all inputs live in first CAP0
    idx = np.zeros((CAPMAX, 4), dtype=np.int64)
    w = np.zeros((CAPMAX, 4), dtype=np.float32)
    b = np.zeros(CAPMAX, dtype=np.float32)
    for i in range(CAP0):
        idx[i] = rng.choice(len(pick), 4, replace=False)
        w[i] = rng.normal(0, .1, 4); b[i] = 0
    alive = np.zeros(CAPMAX, dtype=bool); alive[:CAP0] = True

    def forward():
        s = (X[:, idx] * w[None]).sum(2) + b[None]
        s[:, ~alive] = -np.inf
        out = np.zeros_like(s)
        top = np.argpartition(-s, TOPK - 1, axis=1)[:, :TOPK]
        np.put_along_axis(out, top, 1.0, axis=1)
        return out

    def ev(f):
        proto = np.stack([f[train & (labels == k)].mean(0) for k in range(K)])
        test = ~train
        d = np.abs(f[test][:, None, :] - proto[None]).sum(2)
        pred = d.argmin(1); y = labels[test]; freq = y < FREQ
        return {"overall": float((pred == y).mean()),
                "frequent": float((pred[freq] == y[freq]).mean()),
                "rare": float((pred[~freq] == y[~freq]).mean())}

    f0 = forward(); rows = {"baseline": {"metrics": ev(f0), "alive": int(alive.sum())}}
    usage = f0[train].sum(0)
    order = np.argsort(usage)
    # delete rare 10%: mark dead, no array shift
    alive[order[:CAP0 // 10]] = False
    rows["delete_10_stable"] = {"metrics": ev(forward()), "alive": int(alive.sum())}
    # add 5% mid-frequency new slots
    for i in range(CAP0 // 20):
        slot = CAP0 + i
        idx[slot] = rng.choice(len(pick), 4, replace=False)
        w[slot] = rng.normal(0, .1, 4); b[slot] = 0; alive[slot] = True
    rows["delete_add_stable"] = {"metrics": ev(forward()), "alive": int(alive.sum())}

    out = {"capacity_slots": CAPMAX, "initial_alive": CAP0, "topk": TOPK, "rows": rows}
    with open("results/stable_local.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
