"""Validated search for mid-layer structural plasticity.

3 train / 2 validation / 5 test views per class, random split.
Hidden layer 1,000 units, top-k 200, Ridge readout.  Tune prune/split on
validation, then report held-out test.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge

from .local_patterns import encode
from .model import Local3DNet
from .structural_veryhard import DIMS, K, KS, N, canvas, hard_patterns

H, W = 96, 96
HID = 1000
FAN = 64
IN_DIM = 30000


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= 24) & (xx < 72) & (yy >= 24) & (yy < 72)
    sub = np.sort(rng.choice(np.where(cm)[0], IN_DIM, replace=False))

    images = []; labels = []; split = []
    for k, p in enumerate(pats):
        perm = rng.permutation(10)
        for rank, j in enumerate(perm):
            images.append(canvas(p, rng)); labels.append(k)
            split.append(0 if rank < 3 else (1 if rank < 5 else 2))
    labels = np.array(labels); split = np.array(split)
    train = split == 0; val = split == 1; test = split == 2

    X = np.zeros((len(images), IN_DIM), dtype=bool)
    for i, img in enumerate(images):
        hs = model.forward_numpy(encode(img[None]))
        X[i] = hs[-1][0][sub] > 0

    def run(seed, prune, sfrac, smode, cmode, topk, wscale):
        rng2 = np.random.default_rng(seed)
        hid = 1000
        idx = rng2.integers(0, IN_DIM, (hid, FAN))
        w = rng2.normal(0, .1, (hid, FAN)).astype(np.float32)
        b = np.zeros(hid, dtype=np.float32)

        def hfeat(ii, ww, bb):
            s = (X[:, ii] * ww[None]).sum(2) + bb[None]
            o = np.zeros_like(s)
            top = np.argpartition(-s, topk - 1, axis=1)[:, :topk]
            np.put_along_axis(o, top, 1.0, axis=1)
            return o

        def fit(ii, ww, bb):
            h = hfeat(ii, ww, bb)
            clf = Ridge(alpha=1.0)
            Y = np.eye(K)[labels[train]]
            clf.fit(h[train], Y)
            pv = np.argmax(clf.predict(h[val]), 1)
            pt = np.argmax(clf.predict(h[test]), 1)
            return float((pv == labels[val]).mean()), float((pt == labels[test]).mean()), h, clf.coef_.T

        va, ta, h0, wout = fit(idx, w, b)
        usage = h0[train].sum(0); order = np.argsort(usage)
        nlow = int(hid * prune); nsplit = int(hid * sfrac)
        low = set(order[:nlow].tolist())
        if smode == "high":
            sel = order[-nsplit:] if nsplit else np.array([], dtype=int)
        elif smode == "mid":
            st = (hid - nsplit) // 2; sel = order[st:st + nsplit]
        else:
            cv = h0[train].std(0) / (h0[train].mean(0) + 1e-6)
            cand = np.array([i for i in np.argsort(-cv) if usage[i] >= 3])
            sel = cand[:nsplit]
        sel = np.array([i for i in sel if i not in low])
        keep = [i for i in range(hid) if i not in low and i not in set(sel.tolist())]
        ni = []; nw = []; nb = []
        for i in sel:
            if cmode == "full":
                ni.extend([idx[i].copy(), idx[i].copy()])
                nw.extend([w[i] * .5 * wscale, w[i] * .5 * wscale])
            else:
                ni.extend([np.concatenate([idx[i, :FAN // 2], idx[i, :FAN // 2]]),
                           np.concatenate([idx[i, FAN // 2:], idx[i, FAN // 2:]])])
                nw.extend([np.concatenate([w[i, :FAN // 2], np.zeros(FAN // 2)]) * wscale,
                           np.concatenate([w[i, FAN // 2:], np.zeros(FAN // 2)]) * wscale])
            nb.extend([np.array([b[i] / 2]), np.array([b[i] / 2])])
        if len(ni):
            ii = np.vstack([idx[keep], np.stack(ni)])
            ww = np.vstack([w[keep], np.stack(nw)])
            bb = np.concatenate([b[keep], np.concatenate(nb)])
        else:
            ii, ww, bb = idx[keep], w[keep], b[keep]
        v, t, h1, _ = fit(ii, ww, bb)
        return {"val": float(v), "test": float(t), "n": len(ii)}

    configs = [(0, 0, "high", "half", 200, 1.0),
               (.1, 0, "high", "half", 200, 1.0),
               (0, .1, "high", "half", 200, 1.0),
               (0, .1, "high", "half", 200, 1.0)]
    # random grid
    grng = np.random.default_rng(3)
    while len(configs) < 30:
        configs.append((
            float(grng.choice([0, .05, .1, .2])),
            float(grng.choice([0, .05, .1, .2])),
            str(grng.choice(["high", "mid", "var"])),
            str(grng.choice(["half", "full"])),
            int(grng.choice([150, 200, 250])),
            float(grng.choice([.5, .75, 1.0])),
        ))
    rows = []
    for seed in [0, 1]:
        for c in configs:
            r = run(seed, *c)
            rows.append({"seed": seed, "config": c, **r})
            print(rows[-1], flush=True)
    rows.sort(key=lambda r: (-r["val"], r["n"]))
    best = rows[0]
    # evaluate best test
    out = {"baseline_val": None, "best": best, "rows": rows[:10], "n_configs": len(rows)}
    # baseline val with seed 0, no prune/split
    # reuse run by finding config (0,0)
    for r in rows:
        if r["config"][0] == 0 and r["config"][1] == 0:
            out["baseline_val"] = r["val"]; break
    with open("results/structural_midlayer_search.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

