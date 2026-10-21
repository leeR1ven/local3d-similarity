"""Mid-layer structural plasticity with output-weight splitting.

A fixed 3D encoder produces L5 features.  A 1,000-unit random hidden layer
is followed by a trained linear readout.  High-use hidden units are split
into two children that inherit 2+2 inputs and half of the output weights;
rare units are pruned.  The readout is retrained after each structural edit.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge

from .local_patterns import encode
from .model import Local3DNet
from .structural_veryhard import DIMS, K, KS, N, POP, canvas, hard_patterns

H, W = 96, 96
HID = 1000
FAN = 64
TOPK = 200
IN_DIM = 30000


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    # central L5 feature subset
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= 24) & (xx < 72) & (yy >= 24) & (yy < 72)
    sub = np.sort(rng.choice(np.where(cm)[0], IN_DIM, replace=False))

    images = []; labels = []; split = []
    for k, p in enumerate(pats):
        for j in range(10):
            images.append(canvas(p, rng)); labels.append(k); split.append(j)
    labels = np.array(labels); split = np.array(split)
    X = np.zeros((len(images), IN_DIM), dtype=bool)
    for i, img in enumerate(images):
        hs = model.forward_numpy(encode(img[None]))
        X[i] = hs[-1][0][sub] > 0
    train = split < 5; test = split >= 5

    idx = rng.integers(0, IN_DIM, (HID, FAN))
    w = rng.normal(0, .1, (HID, FAN)).astype(np.float32)
    b = np.zeros(HID, dtype=np.float32)

    def hidden(ii, ww, bb):
        s = (X[:, ii] * ww[None]).sum(2) + bb[None]
        out = np.zeros_like(s)
        top = np.argpartition(-s, TOPK - 1, axis=1)[:, :TOPK]
        np.put_along_axis(out, top, 1.0, axis=1)
        return out

    def fit_eval(ii, ww, bb):
        h = hidden(ii, ww, bb)
        clf = Ridge(alpha=1.0)
        Y = np.eye(K)[labels[train]]
        clf.fit(h[train], Y)
        pred = np.argmax(clf.predict(h[test]), 1)
        return float((pred == labels[test]).mean()), h, clf.coef_.T

    base_acc, h0, wout = fit_eval(idx, w, b)
    usage = h0[train].sum(0); order = np.argsort(usage)
    rows = {"baseline": base_acc}
    low = set(order[:HID // 10].tolist())
    high = order[-HID // 10:][::-1]

    def prune(sel):
        keep = [i for i in range(HID) if i not in set(sel)]
        return idx[keep], w[keep], b[keep], keep

    def split(sel, preserve=False):
        ni = []; nw = []; nb = []; ow = []
        for i in sel:
            if preserve:
                a1 = idx[i].copy(); a2 = idx[i].copy(); w1 = w[i] * .5; w2 = w[i] * .5
            else:
                a1 = np.concatenate([idx[i, :FAN // 2], idx[i, :FAN // 2]])
                a2 = np.concatenate([idx[i, FAN // 2:], idx[i, FAN // 2:]])
                w1 = np.concatenate([w[i, :FAN // 2], np.zeros(FAN // 2)])
                w2 = np.concatenate([w[i, FAN // 2:], np.zeros(FAN // 2)])
            ni.extend([a1, a2]); nw.extend([w1, w2]); nb.extend([b[i] / 2, b[i] / 2])
            ow.extend([wout[i] / 2, wout[i] / 2])
        return np.stack(ni), np.stack(nw), np.array(nb), np.stack(ow)

    # prune only
    pi, pw, pb, _ = prune(low)
    rows["prune_only"] = fit_eval(pi, pw, pb)[0]
    # split only
    si, sw, sb, sow = split(high)
    ii = np.vstack([idx, si]); ww = np.vstack([w, sw]); bb = np.concatenate([b, sb])
    rows["split_only"] = fit_eval(ii, ww, bb)[0]
    # split + prune
    keep = [i for i in range(HID) if i not in low and i not in set(high.tolist())]
    ii = np.vstack([idx[keep], si]); ww = np.vstack([w[keep], sw]); bb = np.concatenate([b[keep], sb])
    rows["split_prune"] = fit_eval(ii, ww, bb)[0]

    out = {"classes": K, "hidden": HID, "topk": TOPK, "rows": rows}
    with open("results/structural_midlayer.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
