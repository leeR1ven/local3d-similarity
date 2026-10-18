"""Opaque local patches: the entire 48x48 patch is identical across backgrounds.

This fixes the earlier flaw where only a shape was identical and the rest of
the 48x48 region was different background.  Retrieval is evaluated per layer
on neurons whose x,y lies inside the patch.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import DIMS, H, KS, N, S, W, X0, Y0, background, encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180


def opaque_patterns(rng):
    pats = []
    for k in range(20):
        p, m = pattern(k, rng)
        full = p.copy()
        bg = rng.integers(0, 10, 3)
        full[m == 0] = bg
        # add a small texture inside the patch so it is a real local image
        noise = rng.integers(-1, 2, full.shape)
        full = np.clip(full + noise, 0, 9)
        pats.append(full.astype(np.int64))
    return pats


def compose(patch, bg):
    img = bg.copy()
    img[Y0:Y0 + S, X0:X0 + S] = patch
    return img


def main():
    rng = np.random.default_rng(0)
    patches = opaque_patterns(rng)
    pixel_proto = np.stack([p.reshape(-1).astype(np.float32) for p in patches])
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    subsets = []
    for dim in DIMS:
        xx = np.repeat(np.arange(dim[0]), dim[1] * dim[2])
        yy = np.tile(np.repeat(np.arange(dim[1]), dim[2]), dim[0])
        cm = (xx >= Y0) & (xx < Y0 + S) & (yy >= X0) & (yy < X0 + S)
        subsets.append(np.sort(rng.choice(np.where(cm)[0], 20000, replace=False)))

    total = 20 * 40
    feats = np.zeros((total, 5, 20000), dtype=bool)
    labels = np.zeros(total, dtype=np.int32)
    is_test = np.zeros(total, dtype=bool)
    pos = 0
    for k, patch in enumerate(patches):
        for j in range(40):
            img = compose(patch, background(rng))
            hs = model.forward_numpy(encode(img[None]))
            for li, h in enumerate(hs):
                feats[pos, li] = h[0][subsets[li]] > 0
            labels[pos] = k
            is_test[pos] = j >= 20
            pos += 1

    rows = []
    tri = np.triu_indices(20, 1)
    pd = np.abs(pixel_proto[:, None] - pixel_proto[None]).sum(-1)
    for li in range(5):
        proto = np.stack([feats[(labels == k) & (~is_test), li].mean(0) for k in range(20)])
        correct = 0
        gaps = []
        for k in range(20):
            for f in feats[(labels == k) & is_test, li]:
                scores = -np.abs(proto - f).sum(1)
                correct += int(np.argmax(scores) == k)
                gaps.append(float(scores[k] - np.delete(scores, k).max()))
        ad = np.abs(proto[:, None] - proto[None]).sum(-1)
        rows.append({
            "layer": li + 1,
            "acc": float(correct / 400),
            "gap": float(np.mean(gaps)),
            "positive": float(np.mean(np.array(gaps) > 0)),
            "pixel_vs_activation_spearman": float(
                np.corrcoef(pd[tri].argsort().argsort(), ad[tri].argsort().argsort())[0, 1]
            ),
        })
    out = {"patterns": 20, "backgrounds": 40, "rows": rows}
    with open("results/local_patterns_opaque.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [r["layer"] for r in rows]
    axes[0].plot(xs, [r["acc"] for r in rows], marker="o", color="#4C78A8")
    axes[0].axhline(0.05, color="#D62728", ls="--", label="chance")
    axes[0].set_ylim(0, 1); axes[0].set_xlabel("层"); axes[0].set_ylabel("检索准确率")
    axes[0].set_title("完整 48×48 局部 patch：逐层检索", fontsize=9); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.2)
    axes[1].plot(xs, [r["pixel_vs_activation_spearman"] for r in rows], marker="o", color="#F58518")
    axes[1].axhline(0, color="#D62728", ls="--")
    axes[1].set_xlabel("层"); axes[1].set_ylabel("patch 像素距离 vs 激活距离 ρ")
    axes[1].set_title("局部 patch 相似性保持", fontsize=9); axes[1].grid(alpha=0.2)
    fig.suptitle("完整局部 patch × 40 背景：前层是否更局部？", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_local_patterns_opaque.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_local_patterns_opaque.png")


if __name__ == "__main__":
    main()
