"""Compare global top-k with pure weight-relative thresholds.

No ranking per image: each layer has a threshold theta * (|b| + sum|w|)
per neuron.  Theta is calibrated once on training images to approximate the
same average sparsity as top-k.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode, pattern
from .model import Local3DNet

H = W = 96
S = 48
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]
K = 20
TARGET = [.5, .4, .3, .2, .1]


def canvas(p, rng):
    bg = rng.integers(0, 10, (24, 24, 3)).astype(np.uint8)
    bg = np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)
    x0 = (W - S) // 2; y0 = (H - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = p
    return bg


def forward_threshold(model, code, theta, upto=None):
    h = code.astype(np.float32); hs = [h]
    n = len(model._np_buffers) if upto is None else upto
    for l in range(n):
        idx, w, b = model._np_buffers[l]
        s = (h[:, idx] * w[None]).sum(-1) + b[None]
        base = np.abs(b) + np.abs(w).sum(1)
        h = (s > theta[l] * base[None]).astype(np.float32)
        hs.append(h)
    return hs


def main():
    rng = np.random.default_rng(0)
    pats = []
    for k in range(K):
        p, m = pattern(k, rng)
        full = p.copy(); full[m == 0] = rng.integers(0, 10, 3)
        pats.append(full.astype(np.int64))
    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        for j in range(10):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < 5)
    labels = np.array(labels); train = np.array(train)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))

    thetas = [0.0] * 4
    for l, (idx, w, b) in enumerate(model._np_buffers):
        base = (np.abs(b) + np.abs(w).sum(1))
        ss = []
        for i in np.where(train)[0]:
            h = encode(images[i][None])
            for ll in range(l):
                s0 = (h[:, model._np_buffers[ll][0]] * model._np_buffers[ll][1][None]).sum(-1) + model._np_buffers[ll][2][None]
                b0 = np.abs(model._np_buffers[ll][2]) + np.abs(model._np_buffers[ll][1]).sum(1)
                h = (s0 > thetas[ll] * b0[None]).astype(np.float32)
            s0 = (h[:, idx] * w[None]).sum(-1) + b[None]
            ss.append(s0[0])
        ss = np.concatenate(ss).reshape(int(train.sum()), -1)
        target = TARGET[l + 1]
        best = (1.0, 0.0)
        for theta in np.linspace(0, 2, 201):
            frac = np.mean(ss > theta * base[None])
            if abs(frac - target) < best[0]:
                best = (abs(frac - target), theta)
        thetas[l] = float(best[1])
    print("thetas", thetas, flush=True)

    def feats(use_topk):
        F = np.zeros((len(images), N[-1]), dtype=bool)
        for i in range(len(images)):
            h = encode(images[i][None])
            if use_topk:
                F[i] = model.forward_numpy(h)[-1][0] > 0
            else:
                F[i] = forward_threshold(model, h, thetas)[-1][0] > 0
        return F

    def ev(F):
        proto = np.stack([F[train & (labels == k)].mean(0) for k in range(K)]).astype(np.float32)
        test = ~train
        Ft = F[test].astype(np.float32)
        d = Ft.sum(1)[:, None] + proto.sum(1)[None, :] - 2 * (Ft @ proto.T)
        pred = d.argmin(1); y = labels[test]
        gap = d[np.arange(len(y)), y] - np.min(np.where(np.eye(K)[y].astype(bool), np.inf, d), 1)
        return {"acc": float((pred == y).mean()), "gap": float(gap.mean()),
                "active": float(F[test].mean())}

    rows = {"topk": ev(feats(True)), "threshold": ev(feats(False)), "theta": thetas}
    with open("results/threshold_activation.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
