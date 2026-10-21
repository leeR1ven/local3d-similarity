"""Scaling study for neuron expansion.

20 classes (10 frequent / 10 rare), 24x24 patches.  Vary capacity N, added
fraction, and top-k; record overall/frequent/rare accuracy.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H, W = 96, 96
CAPS = [1000, 2000, 5000]
ADDS = [0.0, 0.05, 0.10]
KRATIOS = [0.15, 0.24, 0.30]
K = 20
FREQ = 10


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)[:K]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, NLAYER[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    max_pick = np.sort(rng.choice(NLAYER[-1], max(CAPS), replace=False))

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

    rows = {}
    for cap in CAPS:
        pick = max_pick[:cap]
        ii0 = idx0[pick]; ww0 = w0[pick]; bb0 = b0[pick]
        base_feat = feat(ii0, ww0, bb0, max(1, int(cap * .2)))
        usage = base_feat[train].sum(0); order = np.argsort(usage)
        for add in ADDS:
            nadd = int(cap * add)
            sel = order[int(cap * .4):int(cap * .6)][:nadd]
            ni = []; nw = []; nb = []
            for i in sel:
                ni.extend([np.array([ii0[i, 0], ii0[i, 1], ii0[i, 0], ii0[i, 1]]),
                           np.array([ii0[i, 2], ii0[i, 3], ii0[i, 2], ii0[i, 3]])])
                nw.extend([np.array([ww0[i, 0], ww0[i, 1], 0, 0]),
                           np.array([ww0[i, 2], ww0[i, 3], 0, 0])])
                nb.extend([bb0[i] / 2, bb0[i] / 2])
            if ni:
                ii = np.vstack([ii0, np.stack(ni)]); ww = np.vstack([ww0, np.stack(nw)])
                bb = np.concatenate([bb0, np.array(nb)])
            else:
                ii, ww, bb = ii0, ww0, bb0
            for kr in KRATIOS:
                k = max(1, int(len(ii) * kr))
                m = ev(feat(ii, ww, bb, k))
                rows[f"N{cap}_add{add}_k{kr}"] = {"cap": cap, "add": add, "k_ratio": kr,
                                                  "n": len(ii), "k": k, "metrics": m}
                print(cap, add, kr, m, flush=True)
    out = {"rows": rows}
    with open("results/expansion_scaling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
