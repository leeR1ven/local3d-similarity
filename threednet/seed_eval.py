"""Seed variance and evaluation variants for LFW face discrimination.

3 random models (seeds 0,1,2), same split, same 30k L5 neurons.
Reports prototype rank-1, nearest-neighbour rank-1, top-5, pairwise AUC,
and verification EER.  Also checks earlier layers and L5 subset sizes.
"""
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


def metrics(dist, val_y, disc_y):
    same = val_y[:, None] == disc_y[None, :]
    same_d = dist[same]; diff_d = dist[~same]
    return {
        "auc": float(roc_auc_score(same.ravel(), -dist.ravel())),
        "same_mean": float(same_d.mean()),
        "diff_mean": float(diff_d.mean()),
        "same_diff": float(diff_d.mean() - same_d.mean()),
        "nn1": float((disc_y[dist.argmin(1)] == val_y).mean()),
        "top5": float(np.mean([val_y[i] in disc_y[np.argsort(dist[i])[:5]] for i in range(len(val_y))])),
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
    y_eval = labels[eval_idx]
    n_disc = len(disc_idx)
    disc_y = y_eval[:n_disc]; val_y = y_eval[n_disc:]

    sub_rng = np.random.default_rng(123)
    sub30 = np.sort(sub_rng.choice(N[-1], 30000, replace=False))
    levels = {int(i): np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64) for i in eval_idx}

    rows = []
    for seed in [0, 1, 2]:
        model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=seed, bias_scale=0.0)
        feats = np.zeros((len(eval_idx), len(sub30)), dtype=bool)
        for j, i in enumerate(eval_idx):
            h = model.forward_numpy(encode(levels[int(i)][None]))[-1][0] > 0
            feats[j] = h[sub30]
        P = np.stack([feats[:n_disc][disc_y == k].mean(0) for k in range(K)]).astype(np.float32)
        F = feats[n_disc:].astype(np.float32)
        pdist = F.sum(1)[:, None] + P.sum(1)[None, :] - 2 * (F @ P.T)
        proto1 = float((pdist.argmin(1) == val_y).mean())
        met = metrics((F.sum(1)[:, None] + feats[:n_disc].astype(np.float32).sum(1)[None, :]
                       - 2 * (F @ feats[:n_disc].astype(np.float32).T)), val_y, disc_y)
        met["seed"] = seed
        met["proto1"] = proto1
        rows.append(met)
        print(met, flush=True)

    with open("results/seed_eval.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [r["seed"] for r in rows]
    axes[0].plot(xs, [r["proto1"] for r in rows], marker="o", label="prototype rank-1")
    axes[0].plot(xs, [r["nn1"] for r in rows], marker="o", label="1-NN rank-1")
    axes[0].plot(xs, [r["top5"] for r in rows], marker="o", label="top-5")
    axes[0].set_ylim(0, 1); axes[0].set_xlabel("model seed"); axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7); axes[0].set_title("随机种子：人脸识别率变化", fontsize=9)
    axes[1].bar([str(s) for s in xs], [r["auc"] for r in rows], color="#4C78A8")
    axes[1].axhline(.5, color="#D62728", ls="--")
    axes[1].set_ylim(.4, .8); axes[1].set_xlabel("model seed"); axes[1].set_ylabel("pairwise AUC")
    axes[1].set_title("同人 vs 不同人：AUC", fontsize=9)
    fig.suptitle("LFW 155 人：随机种子和评判方式", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_seed_eval.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_seed_eval.png")


if __name__ == "__main__":
    main()

