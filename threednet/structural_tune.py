"""Small adjustments: prune ratio, split ratio, and split middle-frequency units.

The original hypothesis is that *middle-frequency* neurons represent stable
features, not the most frequent ones.  Previous runs split the top 10% most
active units; this run also splits the middle band.
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
from .structural_ablation import eval_variant, make_features, split_children, keep_mid
from .structural_hard2 import DIMS, K, KS, N, POP, canvas, hard_patterns

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180


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
    l4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    base = make_features(l4, idx, w, b, int(POP * .1))
    usage = base[train].sum(0)
    order = np.argsort(usage)
    variants = {"baseline": (idx, w, b, int(POP * .1))}

    def add_prune(name, frac):
        n = int(POP * frac)
        low = set(order[:n].tolist())
        ii, ww, bb = keep_mid(idx, w, b, low, np.array([], dtype=int))
        variants[name] = (ii, ww, bb, int(len(ii) * .1))

    add_prune("prune_1", .01); add_prune("prune_3", .03); add_prune("prune_5", .05)

    def add_split(name, sel, full=False, prune=0.0):
        hi = sel
        if prune:
            low = set(order[:int(POP * prune)].tolist())
            mid = [i for i in range(POP) if i not in low and i not in set(hi.tolist())]
            ii, ww, bb = idx[mid], w[mid], b[mid]
        else:
            ii, ww, bb = idx, w, b
        ni, nw, nb = split_children(idx[hi], w[hi], b[hi], np.arange(len(hi)), preserve=full)
        ii = np.vstack([ii, ni]); ww = np.vstack([ww, nw]); bb = np.concatenate([bb, nb])
        variants[name] = (ii, ww, bb, int(len(ii) * .1))

    add_split("split_mid_5", order[int(POP * .45):int(POP * .50)])
    add_split("split_mid_10", order[int(POP * .40):int(POP * .50)])
    add_split("split_mid_full", order[int(POP * .40):int(POP * .50)], full=True)
    add_split("split_mid_5_prune_1", order[int(POP * .45):int(POP * .50)], prune=.01)
    add_split("split_high_5", order[-int(POP * .05):], full=False)
    add_split("split_high_5_full", order[-int(POP * .05):], full=True)

    rows = {}
    for name, (ii, ww, bb, kk) in variants.items():
        f = make_features(l4, ii, ww, bb, kk)
        acc, gap = eval_variant(f, labels, train)
        rows[name] = {"acc": acc, "gap": gap, "n": int(len(ii)), "topk": kk}
        print(name, rows[name], flush=True)

    with open("results/structural_tune.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = list(rows)
    ax.bar(names, [rows[n]["acc"] for n in names], color="#4C78A8")
    ax.set_ylim(0, 1); ax.set_ylabel("50-class retrieval")
    ax.set_title("prune/split tuning on 24×24 hard task", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig("results/fig_structural_tune.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_structural_tune.png")


if __name__ == "__main__":
    main()
