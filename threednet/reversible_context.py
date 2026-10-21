"""Reversible context adaptation: A -> B -> A without permanent loss.

Two domains of 10 shape classes each.  For each domain we store a temporary
neuron-gain mask (Fisher score on training images).  Switching domains swaps
masks; the base network and all neurons are preserved, so returning to A
restores A's performance exactly.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode, pattern
from .model import Local3DNet

H, W = 96, 96
S = 12
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]
POP = 5000
K = 10


def background(rng):
    bg = rng.integers(0, 10, (24, 24, 3)).astype(np.uint8)
    return np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)


def canvas(p, rng):
    bg = background(rng)
    q = p.copy()
    if rng.random() < .6:
        x = rng.integers(0, S - 4); y = rng.integers(0, S - 4)
        q[y:y + 4, x:x + 4] = rng.integers(0, 10, (4, 4, 3))
    x0 = (W - S) // 2; y0 = (H - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = q
    return bg


def main():
    rng = np.random.default_rng(0)
    pats = []
    for k in range(20):
        p48, m48 = pattern(k, rng)
        p = __import__("cv2").resize(p48, (S, S), interpolation=__import__("cv2").INTER_AREA)
        mask = __import__("cv2").resize(m48.astype(np.uint8) * 255, (S, S), interpolation=__import__("cv2").INTER_NEAREST) > 127
        full = p.copy(); full[~mask] = rng.integers(0, 10, 3)
        pats.append(full.astype(np.int64))
    domains = [pats[:10], pats[10:20]]

    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    pick = np.sort(rng.choice(N[-1], POP, replace=False))

    images = []; labels = []; dom = []; split = []
    for d, ps in enumerate(domains):
        for k, p in enumerate(ps):
            for j in range(20):
                images.append(canvas(p, rng)); labels.append(k); dom.append(d); split.append(j)
    labels = np.array(labels); dom = np.array(dom); split = np.array(split)
    X = np.zeros((len(images), POP), dtype=bool)
    for i, img in enumerate(images):
        X[i] = model.forward_numpy(encode(img[None]))[-1][0][pick] > 0

    def fisher_mask(mask):
        f = X[mask]
        y = labels[mask]
        mean = f.mean(0)
        between = np.zeros(POP); within = np.zeros(POP)
        for k in range(K):
            z = f[y == k]
            between += len(z) * (z.mean(0) - mean) ** 2
            within += ((z - z.mean(0)) ** 2).sum(0)
        score = between / (within + 1e-6)
        keep = np.argsort(-score)[:POP // 2]
        m = np.zeros(POP, dtype=bool); m[keep] = True
        return m

    def eval_domain(d, m):
        tr = (dom == d) & (split < 10)
        te = (dom == d) & (split >= 10)
        proto = np.stack([X[tr & (labels == k)][:, m].mean(0) for k in range(K)])
        f = X[te][:, m]
        pred = np.argmin(np.abs(f[:, None, :] - proto[None]).sum(2), 1)
        return float((pred == labels[te]).mean())

    # baseline all neurons
    base = [eval_domain(d, np.ones(POP, bool)) for d in [0, 1]]
    masks = []
    for d in [0, 1]:
        tr = (dom == d) & (split < 10)
        masks.append(fisher_mask(tr))
    adapted = [[eval_domain(d, masks[d]) for d in [0, 1]] for _ in range(3)]

    out = {
        "baseline_A_B": base,
        "masks_size": [int(m.sum()) for m in masks],
        "sequence": [
            {"context": "A", "acc": eval_domain(0, masks[0])},
            {"context": "B", "acc": eval_domain(1, masks[1])},
            {"context": "A again", "acc": eval_domain(0, masks[0])},
        ],
        "mask_overlap_A_B": int((masks[0] & masks[1]).sum()),
    }
    with open("results/reversible_context.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

