"""Try several neuron-splitting rules on the long-tailed task.

Baseline: 20 classes (10 frequent, 10 rare), N=5,000, k=1,000 fixed.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H, W = 96, 96
POP = 5000
K_ACT = 1000
K = 20
FREQ = 10


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)[:K]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, NLAYER[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(NLAYER[-1], POP, replace=False))
    idx = idx0[pick].copy(); w = w0[pick].copy(); b = b0[pick].copy()

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        n_tr = 20 if k < FREQ else 5
        for j in range(n_tr + 5):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < n_tr)
    labels = np.array(labels); train = np.array(train)
    L4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    def feat(ii, ww, bb):
        out = np.zeros((len(images), len(ii)), dtype=bool)
        for i in range(len(images)):
            s = (L4[i][ii] * ww).sum(1) + bb
            o = np.zeros(len(ii), dtype=bool)
            o[np.argpartition(-s, K_ACT - 1)[:K_ACT]] = True
            out[i] = o
        return out

    def ev(f):
        proto = np.stack([f[train & (labels == k)].mean(0) for k in range(K)])
        test = ~train
        d = np.abs(f[test][:, None, :] - proto[None]).sum(2)
        pred = d.argmin(1); y = labels[test]
        freq = y < FREQ
        return {"overall": float((pred == y).mean()),
                "frequent": float((pred[freq] == y[freq]).mean()),
                "rare": float((pred[~freq] == y[~freq]).mean())}

    f0 = feat(idx, w, b)
    rows = {"baseline": {"metrics": ev(f0), "n": POP}}
    trf = f0[train]; usage = trf.sum(0); order = np.argsort(usage)

    def children(sel, mode, noise=0.0, keep_parent=False):
        ni = []; nw = []; nb = []
        for i in sel:
            if mode == "full":
                a1 = idx[i].copy(); a2 = idx[i].copy()
                w1 = w[i] * .5; w2 = w[i] * .5
            elif mode == "full_noise":
                a1 = idx[i].copy(); a2 = idx[i].copy()
                w1 = w[i] * .5 + rng.normal(0, noise, 4)
                w2 = w[i] * .5 + rng.normal(0, noise, 4)
            else:  # half
                a1 = np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]])
                a2 = np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])
                w1 = np.array([w[i, 0], w[i, 1], 0, 0])
                w2 = np.array([w[i, 2], w[i, 3], 0, 0])
            ni.extend([a1, a2]); nw.extend([w1, w2])
            nb.extend([b[i] / 2, b[i] / 2])
        return np.stack(ni), np.stack(nw), np.array(nb)

    def add(name, sel, mode, noise=0.0, keep_parent=False):
        ni, nw, nb = children(sel, mode, noise)
        if keep_parent:
            ii = np.vstack([idx, ni[::2]]); ww = np.vstack([w, nw[::2]])
            bb = np.concatenate([b, nb[::2]])
        else:
            ii = np.vstack([idx, ni]); ww = np.vstack([w, nw]); bb = np.concatenate([b, nb])
        rows[name] = {"metrics": ev(feat(ii, ww, bb)), "n": len(ii)}

    hi = order[-200:]
    mid = order[2250:2500]
    add("split_high_half", hi, "half")
    add("split_high_full", hi, "full")
    add("split_high_full_noise", hi, "full_noise", noise=.02)
    add("split_mid_half", mid, "half")
    add("split_mid_full", mid, "full")
    add("add_child_high", hi, "full", keep_parent=True)

    with open("results/structural_split_variants.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
