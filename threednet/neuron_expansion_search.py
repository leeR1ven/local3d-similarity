"""Small search around the best neuron-expansion configuration.

Long-tail 20-class task: 10 frequent, 10 rare.  N=5,000.  Search added fraction,
mid-frequency band, child input mode, and top-k.  No independent validation;
the best test row is reported as exploratory, not final.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H, W = 96, 96
POP = 5000
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

    def feat(ii, ww, bb, k):
        out = np.zeros((len(images), len(ii)), dtype=bool)
        for i in range(len(images)):
            s = (L4[i][ii] * ww).sum(1) + bb
            o = np.zeros(len(ii), dtype=bool)
            o[np.argpartition(-s, k - 1)[:k]] = True
            out[i] = o
        return out

    def ev(f):
        proto = np.stack([f[train & (labels == k)].mean(0) for k in range(K)])
        test = ~train
        d = np.abs(f[test][:, None, :] - proto[None]).sum(2)
        pred = d.argmin(1); y = labels[test]; freq = y < FREQ
        return {"overall": float((pred == y).mean()),
                "frequent": float((pred[freq] == y[freq]).mean()),
                "rare": float((pred[~freq] == y[~freq]).mean())}

    base_f = feat(idx, w, b, 1000)
    usage = base_f[train].sum(0); order = np.argsort(usage)
    rows = {"baseline": {"metrics": ev(base_f), "n": POP}}
    best = None
    for frac in [.05, .1, .15, .2]:
        nadd = int(POP * frac)
        for band in [(.4, .6), (.45, .55), (.5, .6)]:
            lo = int(POP * band[0]); hi = int(POP * band[1])
            sel = order[lo:hi][:nadd]
            for mode in ["half", "full"]:
                ni = []; nw = []; nb = []
                for i in sel:
                    if mode == "half":
                        ni.extend([np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]]),
                                   np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])])
                        nw.extend([np.array([w[i, 0], w[i, 1], 0, 0]),
                                   np.array([w[i, 2], w[i, 3], 0, 0])])
                    else:
                        ni.extend([idx[i].copy(), idx[i].copy()])
                        nw.extend([w[i] * .5, w[i] * .5])
                    nb.extend([b[i] / 2, b[i] / 2])
                ii = np.vstack([idx, np.stack(ni)]); ww = np.vstack([w, np.stack(nw)])
                bb = np.concatenate([b, np.array(nb)])
                for k in [800, 1000, 1200]:
                    m = ev(feat(ii, ww, bb, k))
                    key = f"add{frac}_{band[0]}-{band[1]}_{mode}_k{k}"
                    rows[key] = {"metrics": m, "n": len(ii)}
                    if best is None or m["overall"] > rows[best]["metrics"]["overall"]:
                        best = key
    out = {"best": best, "rows": rows}
    with open("results/neuron_expansion_search.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best, "best_row": rows[best]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
