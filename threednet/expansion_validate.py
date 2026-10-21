"""Held-out imbalance: 5 frequent / 15 rare, test the 5% + k=30% rule."""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N as NLAYER, canvas, hard_patterns

H, W = 96, 96
CAP = 5000
K = 20
FREQ = 5


def main():
    rng = np.random.default_rng(1)
    pats = hard_patterns(rng)[:K]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, NLAYER[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(NLAYER[-1], CAP, replace=False))
    ii0 = idx0[pick]; ww0 = w0[pick]; bb0 = b0[pick]

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        n_tr = 20 if k < FREQ else 5
        for j in range(n_tr + 5):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < n_tr)
    labels = np.array(labels); train = np.array(train)
    L4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    def feat(ii, ww, bb):
        k = max(1, int(len(ii) * .3))
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

    base = feat(ii0, ww0, bb0)
    usage = base[train].sum(0); order = np.argsort(usage)
    sel = order[int(CAP * .4):int(CAP * .6)][:CAP // 20]
    ni = []; nw = []; nb = []
    for i in sel:
        ni.extend([np.array([ii0[i, 0], ii0[i, 1], ii0[i, 0], ii0[i, 1]]),
                   np.array([ii0[i, 2], ii0[i, 3], ii0[i, 2], ii0[i, 3]])])
        nw.extend([np.array([ww0[i, 0], ww0[i, 1], 0, 0]), np.array([ww0[i, 2], ww0[i, 3], 0, 0])])
        nb.extend([bb0[i] / 2, bb0[i] / 2])
    exp = feat(np.vstack([ii0, np.stack(ni)]), np.vstack([ww0, np.stack(nw)]),
               np.concatenate([bb0, np.array(nb)]))
    out = {"baseline": ev(base), "expanded": ev(exp), "n_base": CAP, "n_exp": CAP + len(ni)}
    with open("results/expansion_validate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
