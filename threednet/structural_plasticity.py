"""Structural plasticity prototype: prune rare L5 units, split frequent ones.

The 3D encoder is fixed.  A 20,000-neuron L5 subpopulation is tracked over a
stream of repeated local patterns.  Rarely active units are removed; very
frequently active units are split into two children that share the parent's
four inputs 2+2.  Then the same held-out patterns are re-tested.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

H, W = 96, 96
S = 48
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]
POP = 20000
TOPK = 2000


def make_patterns(rng):
    pats = []
    for k in range(20):
        p48, m48 = pattern(k, rng)
        full = p48.copy()
        full[m48 == 0] = rng.integers(0, 10, 3)
        pats.append(full.astype(np.int64))
    return pats


def canvas(p, rng):
    bg = rng.integers(0, 10, (H // 4, W // 4, 3)).astype(np.uint8)
    bg = np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)
    x0 = (W - S) // 2; y0 = (H - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = p
    return bg


def scores(l4, idx, w, b):
    return (l4[idx] * w).sum(1) + b


def topk_act(s, k):
    a = np.zeros(len(s), dtype=bool)
    a[np.argpartition(-s, k - 1)[:k]] = True
    return a


def evaluate(feats, labels, train):
    proto = np.stack([feats[(labels == k) & train].mean(0) for k in range(20)])
    correct = 0; gaps = []
    for k in range(20):
        for f in feats[(labels == k) & (~train)]:
            d = -np.abs(proto - f).sum(1)
            correct += int(np.argmax(d) == k)
            gaps.append(float(d[k] - np.delete(d, k).max()))
    return correct / 100, float(np.mean(gaps))


def main():
    rng = np.random.default_rng(0)
    pats = make_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(N[-1], POP, replace=False))
    idx = idx0[pick].copy(); w = w0[pick].copy(); b = b0[pick].copy()

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        for j in range(10):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < 5)
    labels = np.array(labels); train = np.array(train)

    l4 = []
    for img in images:
        hs = model.forward_numpy(encode(img[None]))
        l4.append(hs[-2][0] > 0)
    l4 = np.stack(l4)

    def features(idx, w, b):
        out = np.zeros((len(images), len(idx)), dtype=bool)
        for i in range(len(images)):
            s = scores(l4[i], idx, w, b)
            out[i] = topk_act(s, TOPK)
        return out

    f0 = features(idx, w, b)
    acc0, gap0 = evaluate(f0, labels, train)
    usage = f0[train].sum(0)
    order = np.argsort(usage)
    n_low = POP // 10; n_high = POP // 10
    low = set(order[:n_low].tolist())
    high = order[-n_high:][::-1]

    keep = [i for i in range(POP) if i not in low and i not in set(high.tolist())]
    new_idx = [idx[i] for i in keep]
    new_w = [w[i] for i in keep]
    new_b = [np.array([b[i]], dtype=np.float32) for i in keep]
    for i in high:
        # two children, 2 active inputs each; pad to 4 with zero-weight copies
        a1 = np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]])
        a2 = np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])
        w1 = np.array([w[i, 0], w[i, 1], 0.0, 0.0], dtype=np.float32)
        w2 = np.array([w[i, 2], w[i, 3], 0.0, 0.0], dtype=np.float32)
        new_idx.extend([a1, a2])
        new_w.extend([w1, w2])
        new_b.extend([np.array([b[i] / 2], dtype=np.float32), np.array([b[i] / 2], dtype=np.float32)])
    new_idx = np.stack(new_idx); new_w = np.stack(new_w); new_b = np.concatenate(new_b)
    # trim to POP if split/prune mismatch
    new_idx = new_idx[:POP]; new_w = new_w[:POP]; new_b = new_b[:POP]
    if len(new_idx) < POP:
        add = POP - len(new_idx)
        new_idx = np.vstack([new_idx, idx[np.arange(add)]])
        new_w = np.vstack([new_w, w[np.arange(add)]])
        new_b = np.concatenate([new_b, b[np.arange(add)]])

    f1 = features(new_idx, new_w, new_b)
    acc1, gap1 = evaluate(f1, labels, train)
    usage1 = f1[train].sum(0)

    out = {
        "population": POP, "topk": TOPK, "patterns": 20,
        "before_acc": acc0, "before_gap": gap0,
        "after_acc": acc1, "after_gap": gap1,
        "low_pruned": n_low, "high_split": n_high,
        "usage_before_mean": float(usage.mean()),
        "usage_after_mean": float(usage1.mean()),
    }
    with open("results/structural_plasticity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["before", "after"], [acc0, acc1], color=["#4C78A8", "#54A24B"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("pattern retrieval")
    axes[0].set_title("prune rare + split frequent", fontsize=9)
    axes[1].hist([usage, usage1], bins=30, label=["before", "after"],
                 color=["#4C78A8", "#54A24B"], alpha=.6)
    axes[1].set_xlabel("activation count on training stream"); axes[1].set_ylabel("units")
    axes[1].legend(fontsize=7); axes[1].set_title("usage distribution", fontsize=9)
    fig.tight_layout(); fig.savefig("results/fig_structural_plasticity.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_structural_plasticity.png")


if __name__ == "__main__":
    main()
