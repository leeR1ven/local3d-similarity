"""Split neurons but keep the absolute top-k fixed.

Key correction: splitting increases N; if k is also increased to keep 10%,
the active fraction stays constant and the intended "harder to activate"
effect disappears.  Here k stays at 2,000 while N grows.
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
from .structural_veryhard import DIMS, K, KS, N, POP, canvas, hard_patterns

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

BASE_K = 2000


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
    base = make_features(l4, idx, w, b, BASE_K)
    usage = base[train].sum(0); order = np.argsort(usage)

    def add(name, ii, ww, bb):
        f = make_features(l4, ii, ww, bb, BASE_K)
        rows[name] = {"acc": eval_variant(f, labels, train)[0],
                      "gap": eval_variant(f, labels, train)[1], "n": len(ii)}

    rows = {}
    add("baseline", idx, w, b)
    low = set(order[:POP // 100].tolist())
    ii, ww, bb = keep_mid(idx, w, b, low, np.array([], dtype=int))
    add("prune_1", ii, ww, bb)
    mid = order[int(POP * .45):int(POP * .50)]
    ni, nw, nb = split_children(idx[mid], w[mid], b[mid], np.arange(len(mid)))
    add("split_mid_5", np.vstack([idx, ni]), np.vstack([w, nw]), np.concatenate([b, nb]))
    add("split_mid_5_full",
        np.vstack([idx, split_children(idx[mid], w[mid], b[mid], np.arange(len(mid)), preserve=True)[0]]),
        np.vstack([w, split_children(idx[mid], w[mid], b[mid], np.arange(len(mid)), preserve=True)[1]]),
        np.concatenate([b, split_children(idx[mid], w[mid], b[mid], np.arange(len(mid)), preserve=True)[2]]))
    keep = [i for i in range(POP) if i not in low and i not in set(mid.tolist())]
    add("split_mid_5_prune_1", np.vstack([idx[keep], ni]), np.vstack([w[keep], nw]),
        np.concatenate([b[keep], nb]))
    high = order[-POP // 20:]
    hi, hw, hb = split_children(idx[high], w[high], b[high], np.arange(len(high)), preserve=True)
    add("split_high_5_full", np.vstack([idx, hi]), np.vstack([w, hw]), np.concatenate([b, hb]))

    with open("results/structural_fixed_k.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = list(rows); ax.bar(names, [rows[n]["acc"] for n in names], color="#4C78A8")
    ax.axhline(rows["baseline"]["acc"], color="#D62728", ls="--", label="baseline")
    ax.set_ylim(0, .3); ax.set_ylabel("50-class retrieval")
    ax.set_title("fixed top-k: split does not increase active fraction", fontsize=10)
    plt.xticks(rotation=30, ha="right"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/fig_structural_fixed_k.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_structural_fixed_k.png")


if __name__ == "__main__":
    main()
