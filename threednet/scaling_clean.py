"""Clean scaling test with constrained, input-driven local weights.

20 complete 32x32 local patches, 6 train + 6 test backgrounds each, 196x108.
Hidden width z is varied 15/30/60/120/240.  Bias is zero and weights are
positive local residuals, so top-k is input-driven rather than a fixed
priority set.
"""
from __future__ import annotations

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import H, W, background, encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

S = 32
TRAIN_BG, TEST_BG = 6, 6


def make_patterns(rng):
    out = []
    for k in range(20):
        p48, m48 = pattern(k, rng)
        p = cv2.resize(p48, (S, S), interpolation=cv2.INTER_AREA)
        mask = m48[:S, :S] > 0
        full = p.copy()
        full[~mask] = rng.integers(0, 10, 3)
        full = np.clip(full + rng.integers(-1, 2, full.shape), 0, 9)
        out.append(full.astype(np.int64))
    return out


def occlude(patch, rng):
    p = patch.copy()
    x = rng.integers(0, S - 8); y = rng.integers(0, S - 8)
    p[y:y + 8, x:x + 8] = rng.integers(0, 10, (8, 8, 3))
    return p


def compose(patch, bg):
    img = bg.copy()
    x0, y0 = (W - S) // 2, (H - S) // 2
    img[y0:y0 + S, x0:x0 + S] = patch
    return img


def to_levels(images):
    """images are already 0..9 levels; never divide by 255 here."""
    return np.clip(images, 0, 9).astype(np.int64)


def main(seed=0):
    rng = np.random.default_rng(seed)
    patches = make_patterns(rng)
    clean = np.stack([p.reshape(-1).astype(np.float32) for p in patches])
    images, labels, is_test = [], [], []
    for k, p in enumerate(patches):
        for j in range(TRAIN_BG + TEST_BG):
            q = occlude(p, rng)
            images.append(compose(q, background(rng)))
            labels.append(k)
            is_test.append(j >= TRAIN_BG)
    images = np.stack(images); labels = np.array(labels); is_test = np.array(is_test)
    levels = to_levels(images)
    x0, y0 = (W - S) // 2, (H - S) // 2

    # raw-pixel template baseline (no mask; occlusion is honest nuisance)
    raw_correct = 0
    for i in np.where(is_test)[0]:
        crop = levels[i, y0:y0 + S, x0:x0 + S].reshape(-1).astype(np.float32)
        d = np.abs(clean - crop).sum(1)
        raw_correct += int(np.argmin(d) == labels[i])
    raw_acc = raw_correct / int(is_test.sum())

    rows = []
    for z in [15, 30, 60, 120, 240]:
        dims = [(H, W, 60), (H, W, z), (H, W, z), (H, W, z), (H, W, z)]
        ns = [d[0] * d[1] * d[2] for d in dims]
        ks = [int(n * s) for n, s in zip(ns, [.5, .4, .3, .2, .1])]
        model = Local3DNet(dims, ks, fan_in=4, alpha=0.01, seed=seed, bias_scale=0.0)
        subs = []
        for dim in dims:
            zz = dim[2]
            xx = np.repeat(np.arange(H), W * zz)
            yy = np.tile(np.repeat(np.arange(W), zz), H)
            cm = (xx >= y0) & (xx < y0 + S) & (yy >= x0) & (yy < x0 + S)
            subs.append(np.where(cm)[0])
        feats = [np.zeros((len(images), len(sub)), dtype=bool) for sub in subs]
        for i in range(len(images)):
            hs = model.forward_numpy(encode(levels[i][None]))
            for li, h in enumerate(hs):
                feats[li][i] = h[0][subs[li]] > 0
        layer_rows = []
        for li in range(5):
            train_f = feats[li][~is_test]
            train_y = labels[~is_test]
            test_f = feats[li][is_test]
            test_y = labels[is_test]
            correct = 0; gaps = []
            for f, y in zip(test_f, test_y):
                d = np.abs(train_f.astype(np.float32) - f.astype(np.float32)).sum(1)
                pred = int(train_y[np.argmin(d)])
                correct += int(pred == y)
                same = d[train_y == y].min()
                other = d[train_y != y].min()
                gaps.append(float(other - same))
            if np.allclose(gaps, 0):
                raise RuntimeError(f"feature collapse: z={z} layer={li+1} gap all zero")
            # null: shuffle test labels, same features/readout
            null_y = np.random.default_rng(123 + li).permutation(test_y)
            null_correct = 0
            for f, y in zip(test_f, null_y):
                d = np.abs(train_f.astype(np.float32) - f.astype(np.float32)).sum(1)
                null_correct += int(int(train_y[np.argmin(d)]) == y)
            null_acc = null_correct / len(test_y)
            proto = np.stack([feats[li][(labels == k) & (~is_test)].mean(0) for k in range(20)])
            ad = np.abs(proto[:, None] - proto[None]).sum(-1)
            tri = np.triu_indices(20, 1)
            pd = np.abs(clean[:, None] - clean[None]).sum(-1)
            layer_rows.append({
                "layer": li + 1,
                "acc": float(correct / (20 * TEST_BG)),
                "gap": float(np.mean(gaps)),
                "positive": float(np.mean(np.array(gaps) > 0)),
                "null_acc": float(null_acc),
                "pixel_vs_activation": float(
                    np.corrcoef(pd[tri].argsort().argsort(), ad[tri].argsort().argsort())[0, 1]
                ),
                "feature_sum": int(feats[li].sum()),
            })
        rows.append({
            "z": z, "l5_neurons": ns[-1], "connections": int(sum(ns[1:]) * 4),
            "layers": layer_rows,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    out_json = "results/scaling_clean.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"patterns": 20, "backgrounds": TRAIN_BG + TEST_BG,
                   "raw_pixel_acc": float(raw_acc), "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = [r["connections"] for r in rows]
    for li in range(5):
        axes[0].plot(xs, [r["layers"][li]["acc"] for r in rows], marker="o", label=f"L{li+1}")
        axes[1].plot(xs, [r["layers"][li]["gap"] for r in rows], marker="o", label=f"L{li+1}")
    axes[0].axhline(0.05, color="#D62728", ls="--", label="chance")
    axes[0].set_xscale("log"); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("connections"); axes[0].set_ylabel("20 pattern retrieval acc")
    axes[0].set_title("模型变大：逐层局部 patch 检索", fontsize=9); axes[0].legend(fontsize=6); axes[0].grid(alpha=0.2)
    axes[1].set_xscale("log"); axes[1].set_xlabel("connections")
    axes[1].set_ylabel("target - max other gap"); axes[1].set_title("模型变大：分离度", fontsize=9)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=0.2)
    fig.suptitle("196×108：z=15/30/60/120/240，20 个 32×32 patch，30% 遮挡", fontsize=12)
    fig_path = "results/fig_scaling_clean.png"
    fig.tight_layout(); fig.savefig(fig_path, bbox_inches="tight"); plt.close(fig)
    print("saved", fig_path)


if __name__ == "__main__":
    main()
