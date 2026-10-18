"""Different-face discrimination by pairwise activation distance.

Instead of closed-set rank-1, decide whether two faces are the same person
(distance <= threshold) or different (distance > threshold).  The threshold
is chosen on discovery pairs; evaluation is on held-out validation pairs.
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


def hamming_matrix(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32)
    return A.sum(1)[:, None] + B.sum(1)[None, :] - 2 * (A @ B.T)


def best_threshold(same, diff):
    ts = np.quantile(np.concatenate([same, diff]), np.linspace(0, 1, 300))
    best = (-1.0, 0.0)
    for t in ts:
        tpr = np.mean(same <= t); tnr = np.mean(diff > t)
        ba = (tpr + tnr) / 2
        if ba > best[0]:
            best = (ba, t)
    return best


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
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= 20) & (xx < 88) & (yy >= 40) & (yy < 156)
    sub = np.sort(rng.choice(np.where(cm)[0], 30000, replace=False))

    feats = np.zeros((len(eval_idx), len(sub)), dtype=bool)
    for j, i in enumerate(eval_idx):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
        feats[j] = h[sub]

    D = feats[:n_disc]; V = feats[n_disc:]
    dd = hamming_matrix(D, D)
    same_mask = disc_y[:, None] == disc_y[None, :]
    tri = np.triu_indices(n_disc, 1)
    train_same = dd[tri][same_mask[tri]]
    train_diff = dd[tri][~same_mask[tri]]
    ba_train, t = best_threshold(train_same, train_diff)

    vd = hamming_matrix(V, D)
    same_val = val_y[:, None] == disc_y[None, :]
    val_same = vd[same_val]; val_diff = vd[~same_val]
    tpr = float(np.mean(val_same <= t)); tnr = float(np.mean(val_diff > t))
    accuracy = float((val_same.size * tpr + val_diff.size * tnr) / (val_same.size + val_diff.size))
    auc = float(roc_auc_score(np.r_[np.zeros(val_same.size), np.ones(val_diff.size)],
                              np.r_[val_same, val_diff]))

    out = {
        "identities": K,
        "train_same_pairs": int(train_same.size), "train_diff_pairs": int(train_diff.size),
        "threshold": float(t), "train_balanced_accuracy": float(ba_train),
        "val_same_pairs": int(val_same.size), "val_diff_pairs": int(val_diff.size),
        "same_accept_rate": tpr, "different_reject_rate": tnr,
        "balanced_accuracy": (tpr + tnr) / 2,
        "pair_accuracy": accuracy,
        "auc": auc,
        "same_mean": float(val_same.mean()), "diff_mean": float(val_diff.mean()),
        "precision_different": float(tnr / max(1e-9, tnr + (1 - tpr))),
        "recall_different": float(tnr),
        "precision_same": float(tpr / max(1e-9, tpr + (1 - tnr))),
        "precision_different_actual": float((val_diff.size * tnr) / max(1e-9, val_diff.size * tnr + val_same.size * (1 - tpr))),
        "precision_same_actual": float((val_same.size * tpr) / max(1e-9, val_same.size * tpr + val_diff.size * (1 - tnr))),
    }
    with open("results/face_pairwise.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["same accept", "different reject"], [tpr, tnr], color=["#54A24B", "#D62728"])
    axes[0].axhline(.5, color="gray", ls="--")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("rate")
    axes[0].set_title(f"pairwise at threshold={t:.0f}", fontsize=9)
    axes[1].hist(val_same, bins=30, alpha=.6, label="same", color="#4C78A8")
    axes[1].hist(val_diff, bins=30, alpha=.5, label="different", color="#F58518")
    axes[1].axvline(t, color="#D62728", ls="--", label="threshold")
    axes[1].set_xlabel("L5 Hamming distance"); axes[1].set_ylabel("count")
    axes[1].legend(fontsize=7); axes[1].set_title("距离分布", fontsize=9)
    fig.suptitle("不同人脸区分：pairwise 判定", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_face_pairwise.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_face_pairwise.png")


if __name__ == "__main__":
    main()


