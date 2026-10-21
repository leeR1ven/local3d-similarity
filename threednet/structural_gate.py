"""Near-threshold controlled task: A/B differ only in a 4x4 block.

With probability DROP the block is replaced by a fixed neutral block, making
the two classes exactly identical for that image.  The same background is
used for both classes, so background cannot leak identity.  Bayes accuracy is
therefore about 1-DROP.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import DIMS, KS, N, scores, topk_act

S = 24
POP = 20000
BASE_K = 2000
DROP = .3
NEUTRAL = np.full((4, 4, 3), 5, dtype=np.int64)


def make_patterns(rng):
    a = rng.integers(0, 10, (S, S, 3))
    b = a.copy()
    b[10:14, 10:14] = rng.integers(0, 10, (4, 4, 3))
    return a, b


def background(rng):
    bg = rng.integers(0, 10, (24, 24, 3)).astype(np.uint8)
    return np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)


def compose(p, bg, drop):
    q = p.copy()
    if drop:
        q[10:14, 10:14] = NEUTRAL
    out = bg.copy()
    x0 = (96 - S) // 2; y0 = (96 - S) // 2
    out[y0:y0 + S, x0:x0 + S] = q
    return out


def features(l4, idx, w, b, k, mask):
    out = np.zeros((int(mask.sum()), len(idx)), dtype=bool)
    j = 0
    for i in np.where(mask)[0]:
        out[j] = topk_act(scores(l4[i], idx, w, b), k); j += 1
    return out


def acc_gap(f, y):
    proto = np.stack([f[y == k].mean(0) for k in range(2)])
    d = np.abs(f[:, None, :] - proto[None]).sum(2)
    pred = d.argmin(1)
    gap = d[np.arange(len(y)), 1 - y] - d[np.arange(len(y)), y]
    return float((pred == y).mean()), float(gap.mean())


def main():
    rng = np.random.default_rng(0)
    a, b = make_patterns(rng)
    images = []; labels = []; split = []
    for j in range(200):
        bg = background(rng)
        drop = rng.random() < DROP
        for k, p in enumerate([a, b]):
            images.append(compose(p, bg, drop)); labels.append(k); split.append(j)
    labels = np.array(labels); split = np.array(split)
    train_m = split < 60; val_m = (split >= 60) & (split < 100); test_m = split >= 100

    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(N[-1], POP, replace=False))
    idx = idx0[pick].copy(); w = w0[pick].copy(); b = b0[pick].copy()
    l4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    bv = acc_gap(features(l4, idx, w, b, BASE_K, val_m), labels[val_m])
    bt = acc_gap(features(l4, idx, w, b, BASE_K, test_m), labels[test_m])
    rows = {"baseline": {"val": bv[0], "test": bt[0], "val_gap": bv[1], "test_gap": bt[1], "n": POP}}
    tr = features(l4, idx, w, b, BASE_K, train_m)
    usage = tr.sum(0); order = np.argsort(usage)

    def add(name, ii, ww, bb, kk=BASE_K):
        v = acc_gap(features(l4, ii, ww, bb, kk, val_m), labels[val_m])
        t = acc_gap(features(l4, ii, ww, bb, kk, test_m), labels[test_m])
        rows[name] = {"val": v[0], "test": t[0], "val_gap": v[1], "test_gap": t[1], "n": len(ii)}

    def keep(sel):
        return np.delete(idx, sel, 0), np.delete(w, sel, 0), np.delete(b, sel, 0)

    def split(sel, full=False):
        ni = []; nw = []; nb = []
        for i in sel:
            if full:
                ni.extend([idx[i].copy(), idx[i].copy()]); nw.extend([w[i] * .5, w[i] * .5])
            else:
                ni.extend([np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]]),
                           np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])])
                nw.extend([np.array([w[i, 0], w[i, 1], 0, 0]), np.array([w[i, 2], w[i, 3], 0, 0])])
            nb.extend([np.array([b[i] / 2]), np.array([b[i] / 2])])
        return np.stack(ni), np.stack(nw), np.concatenate(nb)

    add("prune_1", *keep(order[:200]))
    add("prune_3", *keep(order[:600]))
    mid = order[int(POP * .45):int(POP * .50)]
    ni, nw, nb = split(mid)
    add("split_mid_5", np.vstack([idx, ni]), np.vstack([w, nw]), np.concatenate([b, nb]))
    mid = order[int(POP * .45):int(POP * .50)]
    ni, nw, nb = split(mid, full=True)
    add("split_mid_full", np.vstack([idx, ni]), np.vstack([w, nw]), np.concatenate([b, nb]))
    mid = order[int(POP * .45):int(POP * .50)]
    low = set(order[:200].tolist())
    keep_i = [i for i in range(POP) if i not in low and i not in set(mid.tolist())]
    add("split_mid_5_prune_1", np.vstack([idx[keep_i], ni]), np.vstack([w[keep_i], nw]),
        np.concatenate([b[keep_i], nb]))

    with open("results/structural_gate.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
