"""LFW face test: uniform vs geometry-preserving fovea.

LFW images are 108 high x 196 wide.  The network is built with that
orientation.  The face is already centered by preprocessing, so the fovea
origin is the image center.
"""
from __future__ import annotations

import json

import cv2
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

H, W = 108, 196
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def foveate_face(img):
    base = img.astype(np.float32)
    small = cv2.resize(img.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx + .5 - W / 2) / (W / 2)
    v = (yy + .5 - H / 2) / (H / 2)
    r = np.sqrt(u * u + v * v)
    a = np.clip((r - .45) / (1.0 - .45), 0, 1)
    a = a * a * (3 - 2 * a)
    out = (1 - a[..., None]) * base + a[..., None] * blurred
    out[r > 1] = 0
    return np.clip(out, 0, 9).astype(np.int64)


def main():
    d = np.load("data/lfw_large.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    n5 = N[-1]
    sub = np.sort(rng.choice(n5, 30000, replace=False))

    # 3 discovery + 1 validation per identity (620 images total)
    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        if len(idx) < 4:
            idx = np.resize(idx, 4)
        splits.append((idx[:3], idx[3:4]))

    feats = {name: np.zeros((len(images), len(sub)), dtype=bool) for name in ("uniform", "foveated")}
    for i in range(len(images)):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        f = foveate_face(levels)
        for name, img in (("uniform", levels), ("foveated", f)):
            h = model.forward_numpy(encode(img[None]))[-1][0] > 0
            feats[name][i] = h[sub]

    means = np.stack([images[splits[k][0]].mean(0).reshape(-1) for k in range(K)])
    pix = ((means[:, None] - means[None]) ** 2).sum(-1)
    np.fill_diagonal(pix, np.inf)
    close = pix.min(1) <= np.median(pix.min(1))

    out = {}
    for name in ("uniform", "foveated"):
        val = np.concatenate([s[1] for s in splits])
        proto = np.stack([feats[name][splits[k][0]].mean(0) for k in range(K)])
        F = feats[name][val].astype(np.float32)
        P = proto.astype(np.float32)
        dist = F.sum(1)[:, None] + P.sum(1)[None, :] - 2.0 * (F @ P.T)
        y = labels[val]
        pred = dist.argmin(1)
        out[name] = {
            "rank1": float((pred == y).mean()),
            "rank1_close": float((pred[close[y]] == y[close[y]]).mean()),
            "rank1_distinct": float((pred[~close[y]] == y[~close[y]]).mean()),
            "n_validation": int(len(y)),
        }
    out["identities"] = K
    out["images"] = len(images)
    out["split"] = "3 discovery + 1 validation per identity"
    with open("results/lfw_face_foveated.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    x = np.arange(3); width = .35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [out["uniform"][k] for k in ("rank1", "rank1_close", "rank1_distinct")],
           width, label="uniform", color="#F58518")
    ax.bar(x + width / 2, [out["foveated"][k] for k in ("rank1", "rank1_close", "rank1_distinct")],
           width, label="foveated", color="#4C78A8")
    ax.set_xticks(x); ax.set_xticklabels(["all", "close", "distinct"])
    ax.set_ylim(0, .3); ax.set_ylabel("rank-1"); ax.legend()
    ax.set_title("LFW 108×196: uniform vs foveated", fontsize=10)
    fig.tight_layout(); fig.savefig("results/fig_lfw_face_foveated.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_lfw_face_foveated.png")


if __name__ == "__main__":
    main()
