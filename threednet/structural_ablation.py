"""Separate pruning from splitting on the hard 50-class task.

Variants: baseline, prune only, split only, split+prune, split with preserved
full inputs (no prune).  Active fraction is kept at 10% of the current
population for each variant.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import encode
from .model import Local3DNet
from .structural_hard2 import (
    DIMS, H, K, KS, N, POP, S, W, canvas, hard_patterns, scores, topk_act,
)

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180


def eval_variant(feats, labels, train):
    proto = np.stack([feats[(labels == k) & train].mean(0) for k in range(K)])
    correct = 0; gaps = []
    for k in range(K):
        for f in feats[(labels == k) & (~train)]:
            d = -np.abs(proto - f).sum(1)
            correct += int(np.argmax(d) == k)
            gaps.append(float(d[k] - np.delete(d, k).max()))
    return correct / (K * 5), float(np.mean(gaps))


def make_features(l4, idx, w, b, k):
    out = np.zeros((len(l4), len(idx)), dtype=bool)
    for i in range(len(l4)):
        out[i] = topk_act(scores(l4[i], idx, w, b), k)
    return out


def split_children(idx, w, b, high, preserve=False):
    ni = []; nw = []; nb = []
    for i in high:
        if preserve:
            a1 = idx[i].copy(); a2 = idx[i].copy()
            w1 = w[i] * .5; w2 = w[i] * .5
        else:
            a1 = np.array([idx[i, 0], idx[i, 1], idx[i, 0], idx[i, 1]])
            a2 = np.array([idx[i, 2], idx[i, 3], idx[i, 2], idx[i, 3]])
            w1 = np.array([w[i, 0], w[i, 1], 0, 0], dtype=np.float32)
            w2 = np.array([w[i, 2], w[i, 3], 0, 0], dtype=np.float32)
        ni.extend([a1, a2]); nw.extend([w1, w2])
        nb.extend([np.array([b[i] / 2], dtype=np.float32), np.array([b[i] / 2], dtype=np.float32)])
    return np.stack(ni), np.stack(nw), np.concatenate(nb)


def keep_mid(idx, w, b, low, high):
    keep = [i for i in range(len(idx)) if i not in low and i not in set(high.tolist())]
    return idx[keep], w[keep], np.array([b[i] for i in keep], dtype=np.float32)


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)
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

    base = make_features(l4, idx, w, b, int(POP * .1))
    usage = base[train].sum(0)
    order = np.argsort(usage)
    low = set(order[:POP // 10].tolist())
    high = order[-POP // 10:][::-1]

    variants = {}
    variants["baseline"] = (idx, w, b, int(POP * .1))
    # prune only
    i1, w1, b1 = keep_mid(idx, w, b, low, np.array([], dtype=int))
    variants["prune_only"] = (i1, w1, b1, int(len(i1) * .1))
    # split only
    hi, hw, hb = split_children(idx[high], w[high], b[high], np.arange(len(high)), preserve=False)
    i2 = np.vstack([idx, hi]); w2 = np.vstack([w, hw]); b2 = np.concatenate([b, hb])
    variants["split_only"] = (i2, w2, b2, int(len(i2) * .1))
    # split preserve full inputs
    hi3, hw3, hb3 = split_children(idx[high], w[high], b[high], np.arange(len(high)), preserve=True)
    i3 = np.vstack([idx, hi3]); w3 = np.vstack([w, hw3]); b3 = np.concatenate([b, hb3])
    variants["split_full_inputs"] = (i3, w3, b3, int(len(i3) * .1))
    # split + prune
    i4, w4, b4 = keep_mid(idx, w, b, low, high)
    i4 = np.vstack([i4, hi]); w4 = np.vstack([w4, hw]); b4 = np.concatenate([b4, hb])
    variants["split_plus_prune"] = (i4, w4, b4, int(len(i4) * .1))

    rows = {}
    for name, (ii, ww, bb, kk) in variants.items():
        f = make_features(l4, ii, ww, bb, kk)
        acc, gap = eval_variant(f, labels, train)
        rows[name] = {"n_units": int(len(ii)), "topk": kk, "acc": acc, "gap": gap}
        print(name, rows[name], flush=True)

    with open("results/structural_ablation.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(rows); ax.bar(names, [rows[n]["acc"] for n in names], color="#4C78A8")
    ax.set_ylim(0, 1); ax.set_ylabel("50-class retrieval")
    ax.set_title("prune vs split: hard 50-class task", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig("results/fig_structural_ablation.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_structural_ablation.png")


if __name__ == "__main__":
    main()
