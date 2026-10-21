"""Controlled hard binary task: two patterns differing by one pixel.

Random occlusion sometimes covers the diagnostic pixel, so the baseline is
not 100%.  Structural pruning/splitting is compared on held-out images.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N, canvas, scores, topk_act

S = 12
POP = 20000
BASE_K = 2000


def make_data(rng):
    a = rng.integers(0, 10, (S, S, 3))
    b = a.copy()
    b[6, 6] = (a[6, 6] + 5) % 10
    images = []; labels = []; split = []
    for k, p in enumerate([a, b]):
        for j in range(200):
            images.append(canvas(p, rng)); labels.append(k); split.append(j)
    return np.array(labels), np.array(split), images


def canvas(p, rng):
    bg = rng.integers(0, 10, (24, 24, 3)).astype(np.uint8)
    bg = np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)
    q = p.copy()
    if rng.random() < .7:
        x = rng.integers(0, S - 5); y = rng.integers(0, S - 5)
        q[y:y + 5, x:x + 5] = rng.integers(0, 10, (5, 5, 3))
    x0 = (96 - S) // 2; y0 = (96 - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = q
    return bg


def features(l4, idx, w, b, k, mask):
    out = np.zeros((int(mask.sum()), len(idx)), dtype=bool)
    j = 0
    for i in np.where(mask)[0]:
        out[j] = topk_act(scores(l4[i], idx, w, b), k); j += 1
    return out


def accuracy(f, y):
    proto = np.stack([f[y == k].mean(0) for k in range(2)])
    d = np.abs(f[:, None, :] - proto[None]).sum(2)
    pred = np.argmin(d, 1)
    gap = d[np.arange(len(y)), y] - d[np.arange(len(y)), 1 - y]
    return float((pred == y).mean()), float(gap.mean())


def keep(idx, w, b, sel):
    return np.delete(idx, sel, axis=0), np.delete(w, sel, axis=0), np.delete(b, sel, axis=0)


def split_units(idx, w, b, sel, full=False):
    ni = []; nw = []; nb = []
    for i in sel:
        if full:
            ni.extend([idx[i].copy(), idx[i].copy()])
            nw.extend([w[i] * .5, w[i] * .5])
        else:
            ni.extend([np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]]),
                       np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])])
            nw.extend([np.array([w[i, 0], w[i, 1], 0, 0]),
                       np.array([w[i, 2], w[i, 3], 0, 0])])
        nb.extend([np.array([b[i] / 2]), np.array([b[i] / 2])])
    return np.stack(ni), np.stack(nw), np.concatenate(nb)


def main():
    rng = np.random.default_rng(0)
    labels, split, images = make_data(rng)
    train_m = split < 60; val_m = (split >= 60) & (split < 100); test_m = split >= 100
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(N[-1], POP, replace=False))
    idx = idx0[pick].copy(); w = w0[pick].copy(); b = b0[pick].copy()
    l4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    base_val = features(l4, idx, w, b, BASE_K, val_m)
    base_test = features(l4, idx, w, b, BASE_K, test_m)
    bva = accuracy(base_val, labels[val_m]); bte = accuracy(base_test, labels[test_m])
    rows = {"baseline": {"val": bva[0], "test": bte[0], "val_gap": bva[1], "test_gap": bte[1], "n": POP}}
    tr = features(l4, idx, w, b, BASE_K, train_m)
    usage = tr.sum(0); order = np.argsort(usage)

    def add(name, ii, ww, bb, kk=BASE_K):
        va = accuracy(features(l4, ii, ww, bb, kk, val_m), labels[val_m])
        te = accuracy(features(l4, ii, ww, bb, kk, test_m), labels[test_m])
        rows[name] = {"val": va[0], "test": te[0], "val_gap": va[1], "test_gap": te[1], "n": len(ii)}

    add("prune_1", *keep(idx, w, b, order[:200]))
    add("prune_3", *keep(idx, w, b, order[:600]))
    mid = order[int(POP * .45):int(POP * .50)]
    ni, nw, nb = split_units(idx, w, b, mid, full=False)
    add("split_mid_5", np.vstack([idx, ni]), np.vstack([w, nw]), np.concatenate([b, nb]))
    mid = order[int(POP * .45):int(POP * .50)]
    ni, nw, nb = split_units(idx, w, b, mid, full=True)
    add("split_mid_full", np.vstack([idx, ni]), np.vstack([w, nw]), np.concatenate([b, nb]))
    mid = order[int(POP * .45):int(POP * .50)]
    low = set(order[:200].tolist())
    keep_i = [i for i in range(POP) if i not in low and i not in set(mid.tolist())]
    add("split_mid_5_prune_1", np.vstack([idx[keep_i], ni]), np.vstack([w[keep_i], nw]),
        np.concatenate([b[keep_i], nb]))

    with open("results/structural_binary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

