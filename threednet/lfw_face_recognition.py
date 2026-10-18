"""Large-model LFW face test, stratified by face similarity.

Full L5 population (30k random neurons), no fixed neuron selection.
Reports rank-1 overall, and separately for identities that have a very
similar other face vs identities that are visually distinct.  Also reports
pair verification for similar vs dissimilar negative pairs.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .lfw_large import DIMS, KS, encode
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180


def main():
    d = np.load("data/lfw_large.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    n5 = DIMS[-1][0] * DIMS[-1][1] * DIMS[-1][2]
    sub = np.sort(rng.choice(n5, 30000, replace=False))

    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        n_disc = max(1, int(round(0.6 * len(idx))))
        splits.append((idx[:n_disc], idx[n_disc:]))

    feats = np.zeros((len(images), len(sub)), dtype=bool)
    for i in range(len(images)):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
        feats[i] = h[sub]

    disc_idx = np.concatenate([s[0] for s in splits])
    val_idx = np.concatenate([s[1] for s in splits])
    proto = np.stack([feats[splits[k][0]].mean(0) for k in range(K)])
    F = feats[val_idx].astype(np.float32)
    P = proto.astype(np.float32)
    dist = F.sum(1)[:, None] + P.sum(1)[None, :] - 2.0 * (F @ P.T)

    val_y = labels[val_idx]
    rank1 = float((dist.argmin(1) == val_y).mean())

    means = np.stack([images[splits[k][0]].mean(0).reshape(-1) for k in range(K)])
    pix = ((means[:, None] - means[None]) ** 2).sum(-1)
    np.fill_diagonal(pix, np.inf)
    nearest = pix.min(1)
    close_ids = nearest <= np.median(nearest)
    rank1_close = float((dist.argmin(1)[close_ids[val_y]] == val_y[close_ids[val_y]]).mean())
    rank1_far = float((dist.argmin(1)[~close_ids[val_y]] == val_y[~close_ids[val_y]]).mean())

    pos = dist[np.arange(len(val_y)), val_y]
    threshold = float(np.quantile(pos, 0.9))
    neg_sim, neg_diff = [], []
    for i, k in enumerate(val_y):
        others = rng.choice([j for j in range(K) if j != k], size=12, replace=False)
        for j in others:
            item = (float(dist[i, j]), float(pix[k, j]))
            if pix[k, j] <= np.median(pix[pix < np.inf]):
                neg_sim.append(item)
            else:
                neg_diff.append(item)
    neg_sim = np.array(neg_sim); neg_diff = np.array(neg_diff)
    reject_sim = float((neg_sim[:, 0] > threshold).mean())
    reject_diff = float((neg_diff[:, 0] > threshold).mean())

    out = {
        "identities": K, "images": len(images),
        "rank1": rank1, "chance": 1 / K,
        "rank1_close_ids": rank1_close, "rank1_distinct_ids": rank1_far,
        "close_identities": int(close_ids.sum()),
        "threshold_90pct_same": threshold,
        "similar_negative_pairs": len(neg_sim),
        "dissimilar_negative_pairs": len(neg_diff),
        "true_reject_similar": reject_sim,
        "true_reject_dissimilar": reject_diff,
        "mean_positive_distance": float(pos.mean()),
        "mean_similar_negative_distance": float(neg_sim[:, 0].mean()),
        "mean_dissimilar_negative_distance": float(neg_diff[:, 0].mean()),
    }
    with open("results/lfw_face_recognition.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].bar(["all", "close id", "distinct id"], [rank1, rank1_close, rank1_far],
                color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].axhline(1 / K, color="#D62728", ls="--", label="chance")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("rank-1")
    axes[0].set_title("LFW 人脸 rank-1", fontsize=9); axes[0].legend(fontsize=7)
    axes[1].bar(["similar", "dissimilar"], [reject_sim, reject_diff], color=["#F58518", "#54A24B"])
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("true reject rate")
    axes[1].set_title("不同人脸对：相似 vs 不相似", fontsize=9)
    axes[2].hist([pos, neg_sim[:, 0], neg_diff[:, 0]], bins=30,
                 label=["same", "similar diff", "dissimilar diff"], color=["#4C78A8", "#F58518", "#54A24B"])
    axes[2].set_xlabel("L5 Hamming distance"); axes[2].set_ylabel("count")
    axes[2].set_title("距离分布", fontsize=9); axes[2].legend(fontsize=6)
    fig.suptitle("大模型 LFW：相似人脸更难，不相似人脸应更容易", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_lfw_face_recognition.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_lfw_face_recognition.png")


if __name__ == "__main__":
    main()
