"""Stable-ID layer edits on the 96x96 long-tail task."""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H = W = 96
K = 20
FREQ = 10
PAD = 1.3


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)[:K]
    base = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    base.forward_numpy(np.zeros((1, DIMS[0][0] * DIMS[0][1] * DIMS[0][2]), dtype=np.float32))
    bufs = base._np_buffers
    caps = [int(d[0] * d[1] * d[2] * PAD) for d in DIMS]
    idx = []; w = []; b = []; alive = []
    for li, (ii, ww, bb) in enumerate(bufs):
        n0 = ii.shape[0]; cap = caps[li + 1]
        I = np.zeros((cap, 4), np.int64); Wt = np.zeros((cap, 4), np.float32)
        B = np.zeros(cap, np.float32); A = np.zeros(cap, bool)
        I[:n0] = ii; Wt[:n0] = ww; B[:n0] = bb; A[:n0] = True
        idx.append(I); w.append(Wt); b.append(B); alive.append(A)
    IDX0 = [x.copy() for x in idx]; W0 = [x.copy() for x in w]; B0 = [x.copy() for x in b]

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        n_tr = 20 if k < FREQ else 5
        for j in range(n_tr + 5):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < n_tr)
    labels = np.array(labels); train = np.array(train)
    codes = [encode(im[None]) for im in images]

    def fwd(code):
        h = code
        for li in range(4):
            s = (h[:, idx[li]] * w[li][None]).sum(-1) + b[li][None]
            s[:, ~alive[li]] = -np.inf
            n0 = DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2]
            k = max(1, int(alive[li].sum() * KS[li + 1] / n0))
            o = np.zeros_like(s)
            top = np.argpartition(-s, k - 1, axis=1)[:, :k]
            np.put_along_axis(o, top, 1.0, axis=1)
            h = o
        return h

    def features():
        F = np.zeros((len(codes), caps[-1]), dtype=bool)
        for i, c in enumerate(codes):
            F[i] = fwd(c)[0] > 0
        return F

    def ev(F):
        proto = np.stack([F[train & (labels == k)].mean(0) for k in range(K)]).astype(np.float32)
        test = ~train
        Ft = F[test].astype(np.float32)
        d = Ft.sum(1)[:, None] + proto.sum(1)[None, :] - 2 * (Ft @ proto.T)
        pred = d.argmin(1); y = labels[test]; freq = y < FREQ
        return {"overall": float((pred == y).mean()),
                "frequent": float((pred[freq] == y[freq]).mean()),
                "rare": float((pred[~freq] == y[~freq]).mean())}

    rows = {"baseline": {"metrics": ev(features()), "alive": [int(a.sum()) for a in alive]}}
    for target in range(4):
        idx = [x.copy() for x in IDX0]; w = [x.copy() for x in W0]; b = [x.copy() for x in B0]
        alive = [np.zeros(len(x), bool) for x in idx]
        for li in range(4):
            n0 = DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2]
            alive[li][:n0] = True
        n0 = DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2]
        alive[target][rng.choice(n0, n0 // 10, replace=False)] = False
        for j in range(n0 // 20):
            slot = n0 + j
            idx[target][slot] = rng.choice(DIMS[target][0] * DIMS[target][1] * DIMS[target][2], 4)
            w[target][slot] = rng.normal(0, .05, 4); b[target][slot] = 0; alive[target][slot] = True
            if target + 1 < 4:
                ids = rng.choice(DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2], 4, replace=False)
                idx[target + 1][ids, -1] = slot; w[target + 1][ids, -1] = .05
        rows[f"L{target+2}"] = {"metrics": ev(features()), "alive": [int(a.sum()) for a in alive]}
        print(target + 2, rows[f"L{target+2}"], flush=True)
    with open("results/stable_3d_hard.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
