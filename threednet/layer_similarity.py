"""Layer-by-layer similarity on LFW, and a deeper 8-layer network."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.spatial.distance import cdist

from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

DIMS5 = [(96, 96, 60), (96, 96, 60), (96, 96, 60), (96, 96, 120), (96, 96, 240)]
N5 = [96 * 96 * 60, 96 * 96 * 60, 96 * 96 * 60, 96 * 96 * 120, 96 * 96 * 240]
KS5 = [int(N5[0] * .5), int(N5[1] * .4), int(N5[2] * .3), int(N5[3] * .2), int(N5[4] * .1)]

# 8 layers: same volume, [50,40,30,25,20,15,12,10]% active
DIMS8 = [(96, 96, 60)] * 8
N8 = [96 * 96 * 60] * 8
SP8 = [.5, .4, .3, .25, .2, .15, .12, .10]
KS8 = [int(n * s) for n, s in zip(N8, SP8)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def select_images(n=300):
    d = np.load("data/lfw_centered.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]
    rng = np.random.default_rng(0)
    idx = []
    for k in np.unique(labels):
        ids = np.where(labels == k)[0]
        rng.shuffle(ids)
        idx.extend(ids[:2].tolist())
    idx = np.array(idx); rng.shuffle(idx); idx = idx[:n]
    return images[idx], labels[idx]


def eval_model(model, images, labels, tag):
    rng = np.random.default_rng(1)
    levels = np.round(images.astype(np.float32) / 255.0 * 9).astype(np.int64)
    flat = levels.reshape(len(images), -1).astype(np.float32)
    d_img = cdist(flat, flat, metric="cityblock")
    tri = np.triu_indices(len(images), 1)
    same = labels[:, None] == labels[None, :]
    same_pairs = same[tri]
    rows = []
    subsets = {}
    for li in range(len(model.dims)):
        n = model.dims[li][0] * model.dims[li][1] * model.dims[li][2]
        sub = rng.choice(n, min(20000, n), replace=False)
        subsets[li] = np.sort(sub)
    # collect activations layer by layer with one forward per image
    feats = [[] for _ in model.dims]
    for im in levels:
        hs = model.forward_numpy(encode(im[None]))
        for li, h in enumerate(hs):
            feats[li].append(h[0][subsets[li]])
    for li, fl in enumerate(feats):
        f = np.stack(fl).astype(np.float32)
        d_act = cdist(f, f, metric="hamming") * f.shape[1]
        d_act = d_act[tri]
        rho = float(np.corrcoef(d_img[tri].argsort().argsort(), d_act.argsort().argsort())[0, 1])
        rows.append({
            "layer": li + 1,
            "neurons": int(len(subsets[li])),
            "spearman": rho,
            "same_identity_mean": float(d_act[same_pairs].mean()),
            "different_identity_mean": float(d_act[~same_pairs].mean()),
            "same_ratio": float(d_act[same_pairs].mean() / d_act[~same_pairs].mean()),
        })
    return rows


def main():
    images, labels = select_images(300)
    rows5 = eval_model(Local3DNet(DIMS5, KS5, fan_in=4, alpha=0.01, seed=0), images, labels, "5")
    rows8 = eval_model(Local3DNet(DIMS8, KS8, fan_in=4, alpha=0.01, seed=1), images, labels, "8")
    out = {"n_images": len(images), "five_layer": rows5, "eight_layer": rows8}
    with open("results/layer_similarity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    xs = [r["layer"] for r in rows5]
    ys = [r["spearman"] for r in rows5]
    ax.plot(xs, ys, marker="o", color="#4C78A8")
    ax.set_ylim(0, 1.02); ax.set_xticks(xs)
    ax.set_xlabel("层"); ax.set_ylabel("像素距离 vs 激活距离 Spearman")
    ax.set_title("5 层网络：逐层相似性", fontsize=9)
    ax.grid(alpha=0.2)
    ax = axes[1]
    xs = [r["layer"] for r in rows8]
    ys = [r["spearman"] for r in rows8]
    ax.plot(xs, ys, marker="o", color="#F58518")
    ax.set_ylim(0, 1.02); ax.set_xticks(xs)
    ax.set_xlabel("层"); ax.set_ylabel("像素距离 vs 激活距离 Spearman")
    ax.set_title("8 层网络：逐层相似性", fontsize=9)
    ax.grid(alpha=0.2)
    fig.suptitle("不只看第5层：L2–L8 都保持画面相似性吗？", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/fig_layer_similarity.png", bbox_inches="tight")
    plt.close(fig)
    print("saved results/fig_layer_similarity.png")


if __name__ == "__main__":
    main()
