"""Validate margin effects across three random image seeds."""
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


def main():
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, DIMS[0][0] * DIMS[0][1] * DIMS[0][2]), dtype=np.float32))
    bufs = model._np_buffers
    caps = [int(d[0] * d[1] * d[2] * PAD) for d in DIMS]
    idx0 = []; w0 = []; b0 = []; a0 = []
    for li, (ii, ww, bb) in enumerate(bufs):
        n0 = ii.shape[0]; cap = caps[li + 1]
        I = np.zeros((cap, 4), np.int64); Wt = np.zeros((cap, 4), np.float32)
        B = np.zeros(cap, np.float32); A = np.zeros(cap, bool)
        I[:n0] = ii; Wt[:n0] = ww; B[:n0] = bb; A[:n0] = True
        idx0.append(I); w0.append(Wt); b0.append(B); a0.append(A)
    sub = np.sort(np.random.default_rng(999).choice(caps[-1], SUB, replace=False))

    def fwd(code, idx, w, b, alive):
        h = code; hs = []
        for li in range(4):
            s = (h[:, idx[li]] * w[li][None]).sum(-1) + b[li][None]
            s[:, ~alive[li]] = -np.inf
            n0 = DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2]
            k = max(1, int(alive[li].sum() * KS[li + 1] / n0))
            o = np.zeros_like(s); top = np.argpartition(-s, k - 1, axis=1)[:, :k]
            np.put_along_axis(o, top, 1.0, axis=1); h = o; hs.append(h)
        return hs

    def feats(codes, idx, w, b, alive):
        F = np.zeros((len(codes), SUB), bool)
        for i, c in enumerate(codes):
            F[i] = fwd(c, idx, w, b, alive)[-1][0][sub] > 0
        return F

    def metric(F, labels):
        d = F.sum(1)[:, None] + F.sum(1)[None, :] - 2 * (F.astype(np.float32) @ F.T.astype(np.float32))
        same = labels[:, None] == labels[None, :]; tri = np.triu_indices(len(labels), 1)
        sd = d[tri][same[tri]]; dd = d[tri][~same[tri]]
        return {"ratio": float(sd.mean() / dd.mean()), "min_diff": float(dd.min()),
                "same": float(sd.mean()), "diff": float(dd.mean()),
                "auc": float(roc_auc_score(same[tri], -d[tri]))}

    rows = {}
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        imgs = []; labels = []
        for k in range(100):
            x = rng.integers(0, 10, (H, W, 3)); imgs.append(x); labels.append(k)
            y = x.copy(); yy = rng.integers(0, H - 6); xx = rng.integers(0, W - 6)
            y[yy:yy + 6, xx:xx + 6] = rng.integers(0, 10, (6, 6, 3))
            imgs.append(y); labels.append(k)
        labels = np.array(labels); codes = [encode(x[None]) for x in imgs]
        base_hs = [fwd(c, idx0, w0, b0, a0) for c in codes]
        rows[seed] = {"baseline": metric(feats(codes, idx0, w0, b0, a0), labels)}
        for target in [1, 3]:
            for frac in [.05, .1]:
                idx = [x.copy() for x in idx0]; w = [x.copy() for x in w0]; b = [x.copy() for x in b0]
                alive = [x.copy() for x in a0]
                n0 = DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2]
                U = np.stack([h[target][0] for h in base_hs]).sum(0)[:n0]
                order = np.argsort(U)
                nadd = max(1, int(n0 * frac))
                for j, parent in enumerate(order[int(n0 * .4):int(n0 * .5)][:nadd]):
                    slot = n0 + j
                    idx[target][slot] = rng.choice(DIMS[target][0] * DIMS[target][1] * DIMS[target][2], 4)
                    w[target][slot] = rng.normal(0, .05, 4); b[target][slot] = 0; alive[target][slot] = True
                    if target + 1 < 4:
                        ids = rng.choice(DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2], 4, replace=False)
                        idx[target + 1][ids, -1] = slot; w[target + 1][ids, -1] = .05
                key = f"L{target+2}_add{frac}"
                rows[seed][key] = metric(feats(codes, idx, w, b, alive), labels)
        print(seed, rows[seed], flush=True)
    with open("results/margin_validate.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
