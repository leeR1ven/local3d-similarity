"""Confirm the capacity-5000 split-mid result across L5 subsets."""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_gate import background, compose, make_patterns
from .structural_hard2 import DIMS, KS, N as NLAYER

CAP = 5000


def main():
    rng = np.random.default_rng(0)
    a, b = make_patterns(rng)
    images = []; labels = []; split = []
    for j in range(200):
        bg = background(rng); drop = rng.random() < .3
        for k, p in enumerate([a, b]):
            images.append(compose(p, bg, drop)); labels.append(k); split.append(j)
    labels = np.array(labels); split = np.array(split)
    train = split < 60; val = (split >= 60) & (split < 100); test = split >= 100

    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, NLAYER[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    L4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    def feat(ii, ww, bb, mask):
        k = max(1, len(ii) // 10)
        out = np.zeros((int(mask.sum()), len(ii)), dtype=bool)
        j = 0
        for i in np.where(mask)[0]:
            s = (L4[i][ii] * ww).sum(1) + bb
            o = np.zeros(len(ii), dtype=bool)
            o[np.argpartition(-s, k - 1)[:k]] = True
            out[j] = o; j += 1
        return out

    def ev(f, y):
        p = np.stack([f[y == k].mean(0) for k in range(2)])
        d = np.abs(f[:, None, :] - p[None]).sum(2)
        return float((d.argmin(1) == y).mean())

    rows = {}
    for seed in [0, 1, 2]:
        r = np.random.default_rng(seed)
        pick = np.sort(r.choice(NLAYER[-1], CAP, replace=False))
        ii = idx0[pick]; ww = w0[pick]; bb = b0[pick]
        base = ev(feat(ii, ww, bb, test), labels[test])
        tr = feat(ii, ww, bb, train); usage = tr.sum(0); order = np.argsort(usage)
        mid = order[CAP // 2 - 250:CAP // 2 + 250]
        ni = []; nw = []; nb = []
        for i in mid:
            ni.extend([np.array([ii[i, 0], ii[i, 1], ii[i, 0], ii[i, 1]]),
                       np.array([ii[i, 2], ii[i, 3], ii[i, 2], ii[i, 3]])])
            nw.extend([np.array([ww[i, 0], ww[i, 1], 0, 0]),
                       np.array([ww[i, 2], ww[i, 3], 0, 0])])
            nb.extend([bb[i] / 2, bb[i] / 2])
        split_acc = ev(feat(np.vstack([ii, np.stack(ni)]), np.vstack([ww, np.stack(nw)]),
                            np.concatenate([bb, np.array(nb)]), test), labels[test])
        rows[seed] = {"baseline": base, "split_mid": split_acc}
        print(seed, rows[seed], flush=True)
    with open("results/structural_capacity_confirm.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
