"""Margin-based structural study on random images.

Top-1 is already 100%; instead measure:
- same-base perturbation distance,
- different-base distance,
- ratio and pairwise AUC.
Then add neurons in one 3D layer and see if separation strengthens.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from .local_patterns import encode
from .model import Local3DNet

H = W = 32
DIMS = [(H, W, 60), (H, W, 30), (H, W, 30), (H, W, 60), (H, W, 120)]
KS = [int(H * W * 60 * .5), int(H * W * 30 * .4), int(H * W * 30 * .3),
      int(H * W * 60 * .2), int(H * W * 120 * .1)]
PAD = 1.5
SUB = 30000
rng = np.random.default_rng(0)


def build_stable(model):
    bufs = model._np_buffers
    caps = [int(d[0] * d[1] * d[2] * PAD) for d in DIMS]
    idx = []; w = []; b = []; alive = []
    for li, (ii, ww, bb) in enumerate(bufs):
        n0 = ii.shape[0]; cap = caps[li + 1]
        I = np.zeros((cap, 4), np.int64); Wt = np.zeros((cap, 4), np.float32)
        B = np.zeros(cap, np.float32); A = np.zeros(cap, bool)
        I[:n0] = ii; Wt[:n0] = ww; B[:n0] = bb; A[:n0] = True
        idx.append(I); w.append(Wt); b.append(B); alive.append(A)
    return idx, w, b, alive, caps


def fwd(code, idx, w, b, alive):
    h = code; hs = []
    for li in range(4):
        s = (h[:, idx[li]] * w[li][None]).sum(-1) + b[li][None]
        s[:, ~alive[li]] = -np.inf
        n0 = DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2]
        k = max(1, int(alive[li].sum() * KS[li + 1] / n0))
        o = np.zeros_like(s)
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        np.put_along_axis(o, top, 1.0, axis=1)
        h = o; hs.append(h)
    return hs


def main():
    base = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    base.forward_numpy(np.zeros((1, DIMS[0][0] * DIMS[0][1] * DIMS[0][2]), dtype=np.float32))
    base_idx, base_w, base_b, base_alive, caps = build_stable(base)

    imgs = []
    labels = []
    for k in range(100):
        x = rng.integers(0, 10, (H, W, 3))
        imgs.append(x); labels.append(k)
        y = x.copy(); yy = rng.integers(0, H - 6); xx = rng.integers(0, W - 6)
        y[yy:yy + 6, xx:xx + 6] = rng.integers(0, 10, (6, 6, 3))
        imgs.append(y); labels.append(k)
    labels = np.array(labels); codes = [encode(x[None]) for x in imgs]
    sub = np.sort(rng.choice(caps[-1], SUB, replace=False))

    def metrics(F):
        d = F.sum(1)[:, None] + F.sum(1)[None, :] - 2 * (F.astype(np.float32) @ F.T.astype(np.float32))
        same = labels[:, None] == labels[None, :]
        tri = np.triu_indices(len(labels), 1)
        sd = d[tri][same[tri]]; dd = d[tri][~same[tri]]
        return {"same": float(sd.mean()), "diff": float(dd.mean()),
                "ratio": float(sd.mean() / dd.mean()),
                "min_diff": float(dd.min()),
                "auc": float(roc_auc_score(same[tri], -d[tri]))}

    def features(idx, w, b, alive):
        F = np.zeros((len(codes), len(sub)), dtype=bool)
        for i, c in enumerate(codes):
            F[i] = fwd(c, idx, w, b, alive)[-1][0][sub] > 0
        return F

    rows = {"baseline": metrics(features(base_idx, base_w, base_b, base_alive))}
    base_hs = [fwd(c, base_idx, base_w, base_b, base_alive) for c in codes]
    for target in range(4):
        idx = [x.copy() for x in base_idx]; w = [x.copy() for x in base_w]; b = [x.copy() for x in base_b]
        alive = [x.copy() for x in base_alive]
        n0 = DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2]
        # usage at target layer
        U = np.stack([h[target][0] for h in base_hs]).sum(0)[:n0]
        order = np.argsort(U)
        sel = order[int(n0 * .4):int(n0 * .5)]
        # add 5% new slots and rewire downstream
        for j, parent in enumerate(sel[:n0 // 20]):
            slot = n0 + j
            idx[target][slot] = rng.choice(DIMS[target][0] * DIMS[target][1] * DIMS[target][2], 4)
            w[target][slot] = rng.normal(0, .05, 4); b[target][slot] = 0
            alive[target][slot] = True
            if target + 1 < 4:
                ids = rng.choice(DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2], 4, replace=False)
                idx[target + 1][ids, -1] = slot; w[target + 1][ids, -1] = .05
        rows[f"add_L{target+2}"] = metrics(features(idx, w, b, alive))
        print(target + 2, rows[f"add_L{target+2}"], flush=True)

    with open("results/discrimination_margin.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
