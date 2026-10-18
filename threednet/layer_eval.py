"""Which layer / how many L5 neurons best discriminates LFW faces?"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score

from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

H, W = 108, 196
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def evaluate(feats, n_disc, val_y, disc_y):
    D = feats[:n_disc].astype(np.float32)
    F = feats[n_disc:].astype(np.float32)
    dist_nn = F.sum(1)[:, None] + D.sum(1)[None, :] - 2 * (F @ D.T)
    same = val_y[:, None] == disc_y[None, :]
    P = np.stack([feats[:n_disc][disc_y == k].mean(0) for k in range(len(np.unique(disc_y)))]).astype(np.float32)
    dist_proto = F.sum(1)[:, None] + P.sum(1)[None, :] - 2 * (F @ P.T)
    return {
        "proto1": float((dist_proto.argmin(1) == val_y).mean()),
        "top5": float(np.mean([val_y[i] in disc_y[np.argsort(dist_nn[i])[:5]] for i in range(len(val_y))])),
        "auc": float(roc_auc_score(same.ravel(), -dist_nn.ravel())),
        "same_mean": float(dist_nn[same].mean()),
        "diff_mean": float(dist_nn[~same].mean()),
    }


def main():
    d = np.load("data/lfw_large.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        splits.append((idx[:2], idx[2:3]))
    disc_idx = np.concatenate([s[0] for s in splits])
    val_idx = np.concatenate([s[1] for s in splits])
    eval_idx = np.concatenate([disc_idx, val_idx])
    y = labels[eval_idx]
    n_disc = len(disc_idx)
    disc_y, val_y = y[:n_disc], y[n_disc:]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    rng2 = np.random.default_rng(9)
    subs = []
    for dim in DIMS:
        n = dim[0] * dim[1] * dim[2]
        subs.append(np.sort(rng2.choice(n, 60000, replace=False)))

    feats = np.zeros((len(eval_idx), 5, 60000), dtype=bool)
    for j, i in enumerate(eval_idx):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        hs = model.forward_numpy(encode(levels[None]))
        for li, h in enumerate(hs):
            feats[j, li] = h[0][subs[li]] > 0
    rows = []
    for li in range(5):
        r = evaluate(feats[:, li], n_disc, val_y, disc_y)
        r["layer"] = li + 1
        rows.append(r)
        print(r, flush=True)

    # L5 subset sizes
    sub_rows = []
    for k in [5000, 15000, 30000, 60000]:
        r = evaluate(feats[:, 4, :k], n_disc, val_y, disc_y)
        r["n_neurons"] = k
        sub_rows.append(r)
        print(r, flush=True)

    out = {"layers": rows, "l5_subset_sizes": sub_rows}
    with open("results/layer_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [r["layer"] for r in rows]
    axes[0].plot(xs, [r["proto1"] for r in rows], marker="o", label="prototype rank-1")
    axes[0].plot(xs, [r["top5"] for r in rows], marker="o", label="top-5")
    axes[0].set_ylim(0, .4); axes[0].set_xlabel("layer"); axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7); axes[0].set_title("逐层 LFW 区分", fontsize=9)
    axes[1].plot([r["n_neurons"] for r in sub_rows], [r["proto1"] for r in sub_rows], marker="o")
    axes[1].set_xscale("log"); axes[1].set_xlabel("L5 neurons"); axes[1].set_ylabel("prototype rank-1")
    axes[1].set_title("L5 神经元数量", fontsize=9)
    fig.suptitle("LFW 155 人：层数和神经元数量", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_layer_eval.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_layer_eval.png")


if __name__ == "__main__":
    main()
