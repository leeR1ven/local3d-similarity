"""Do similar LFW faces activate similar L5 neurons?  No identity classification.

Uses a 20,000-neuron random subset of L5 to estimate Hamming distances so that
all pairwise distances for 400 images fit in memory.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

DIMS = [(96, 96, 60), (96, 96, 60), (96, 96, 60), (96, 96, 120), (96, 96, 240)]
N = [96 * 96 * 60, 96 * 96 * 60, 96 * 96 * 60, 96 * 96 * 120, 96 * 96 * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def main():
    d = np.load("data/lfw_centered.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]
    rng = np.random.default_rng(0)
    idx = []
    for k in np.unique(labels):
        ids = np.where(labels == k)[0]
        rng.shuffle(ids)
        idx.extend(ids[:3].tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    idx = idx[:400]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    n5 = N[-1]
    sub = rng.choice(n5, 20000, replace=False)
    sub.sort()
    feats = []
    levels_all = []
    for i in idx:
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
        feats.append(h[sub])
        levels_all.append(levels)
    feats = np.stack(feats).astype(np.float32)
    levels_all = np.stack(levels_all).astype(np.float32)
    # pairwise L1 on levels, Hamming on the 20k-neuron subset
    d_act = (feats[:, None, :] != feats[None, :, :]).sum(-1)
    d_img = np.abs(levels_all[:, None] - levels_all[None]).sum(-1).sum(-1).sum(-1)
    tri = np.triu_indices(len(idx), 1)
    x = d_img[tri]; y = d_act[tri]
    rho = float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
    same = labels[idx][:, None] == labels[idx][None, :]
    same_pairs = same[tri]
    out = {
        "n_images": int(len(idx)),
        "n_l5_subset": int(len(sub)),
        "pixel_vs_activation_spearman": rho,
        "same_identity_activation_mean": float(y[same_pairs].mean()),
        "different_identity_activation_mean": float(y[~same_pairs].mean()),
        "same_identity_pixel_mean": float(x[same_pairs].mean()),
        "different_identity_pixel_mean": float(x[~same_pairs].mean()),
        "same_identity_pixel_ratio": float(x[same_pairs].mean() / x[~same_pairs].mean()),
        "same_identity_ratio": float(y[same_pairs].mean() / y[~same_pairs].mean()),
    }
    with open("results/lfw_similarity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.hexbin(x, y, gridsize=45, cmap="Blues", mincnt=1)
    ax.set_xlabel("输入像素 L1 距离")
    ax.set_ylabel("L5 Hamming 距离（2 万神经元子集）")
    ax.set_title(f"输入越相似，L5 激活越相似：ρ={rho:.3f}", fontsize=9)
    ax = axes[1]
    ax.bar(["同身份", "不同身份"], [out["same_identity_activation_mean"], out["different_identity_activation_mean"]],
           color=["#4C78A8", "#F58518"], width=0.55)
    ax.set_ylabel("L5 Hamming 距离")
    ax.set_title("同身份只是略近，不是分类标准", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("LFW 400 张：相似画面 → 相似激活（不是身份分类）", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/fig_lfw_similarity.png", bbox_inches="tight")
    plt.close(fig)
    print("saved results/fig_lfw_similarity.png")


if __name__ == "__main__":
    main()
