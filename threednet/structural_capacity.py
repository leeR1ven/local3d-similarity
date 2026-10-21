"""Capacity-limited structural test using actual L5 input weights.

A small random subset of L5 neurons is available (capacity N).  Top-k is 10%
of the current population.  Prune/split are compared at each capacity.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_gate import background, compose, make_patterns
from .structural_hard2 import DIMS, KS, N as NLAYER

CAPS = [100, 200, 500, 1000, 2000, 5000]
POP_MAX = 5000


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
    pick = np.sort(rng.choice(NLAYER[-1], POP_MAX, replace=False))
    idx_all = idx0[pick].copy(); w_all = w0[pick].copy(); b_all = b0[pick].copy()

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
        pred = d.argmin(1)
        return float((pred == y).mean())

    def split(ii, ww, bb, sel):
        ni = []; nw = []; nb = []
        for i in sel:
            ni.extend([np.array([ii[i, 0], ii[i, 1], ii[i, 0], ii[i, 1]]),
                       np.array([ii[i, 2], ii[i, 3], ii[i, 2], ii[i, 3]])])
            nw.extend([np.array([ww[i, 0], ww[i, 1], 0, 0]),
                       np.array([ww[i, 2], ww[i, 3], 0, 0])])
            nb.extend([bb[i] / 2, bb[i] / 2])
        return np.stack(ni), np.stack(nw), np.array(nb)

    rows = {}
    for cap in CAPS:
        ii = idx_all[:cap]; ww = w_all[:cap]; bb = b_all[:cap]
        row = {"baseline": {"val": ev(feat(ii, ww, bb, val), labels[val]),
                            "test": ev(feat(ii, ww, bb, test), labels[test]), "n": cap}}
        tr = feat(ii, ww, bb, train); usage = tr.sum(0); order = np.argsort(usage)
        n = max(1, cap // 10)
        keep = order[n:]
        row["prune_10"] = {"val": ev(feat(ii[keep], ww[keep], bb[keep], val), labels[val]),
                           "test": ev(feat(ii[keep], ww[keep], bb[keep], test), labels[test]),
                           "n": len(keep)}
        sel = order[-n:]
        ni, nw, nb = split(ii, ww, bb, sel)
        row["split_high_10"] = {"val": ev(feat(np.vstack([ii, ni]), np.vstack([ww, nw]), np.concatenate([bb, nb]), val), labels[val]),
                                "test": ev(feat(np.vstack([ii, ni]), np.vstack([ww, nw]), np.concatenate([bb, nb]), test), labels[test]),
                                "n": cap + len(ni)}
        mid = order[cap // 2 - n // 2:cap // 2 + n // 2]
        ni, nw, nb = split(ii, ww, bb, mid)
        row["split_mid_10"] = {"val": ev(feat(np.vstack([ii, ni]), np.vstack([ww, nw]), np.concatenate([bb, nb]), val), labels[val]),
                               "test": ev(feat(np.vstack([ii, ni]), np.vstack([ww, nw]), np.concatenate([bb, nb]), test), labels[test]),
                               "n": cap + len(ni)}
        rows[cap] = row
        print(cap, row, flush=True)

    with open("results/structural_capacity.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
