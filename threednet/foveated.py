"""Circular foveated input: more pixels near the center, fewer at the edge.

Compares uniform resize vs log-polar-like foveation on 10 local patterns.
The whole circular field is kept; the center is magnified, the periphery is
compressed, and the corners are masked out.
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

SRC = 256
OUT = 160
S = 64
DIMS = [(OUT, OUT, 60), (OUT, OUT, 60), (OUT, OUT, 60), (OUT, OUT, 120), (OUT, OUT, 240)]
N = [OUT * OUT * 60, OUT * OUT * 60, OUT * OUT * 60, OUT * OUT * 120, OUT * OUT * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def make_patterns(rng):
    out = []
    for k in range(10):
        p48, m48 = pattern(k, rng)
        p = cv2.resize(p48, (S, S), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(m48.astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_NEAREST) > 127
        full = p.copy()
        full[~mask] = rng.integers(0, 10, 3)
        out.append(full.astype(np.int64))
    return out


def source_image(p, rng):
    small = rng.integers(0, 10, (32, 32, 3)).astype(np.uint8)
    bg = cv2.resize(small, (SRC, SRC), interpolation=cv2.INTER_CUBIC).astype(np.int64)
    x = (SRC - S) // 2
    bg[x:x + S, x:x + S] = p
    return bg


def uniform(img):
    return cv2.resize(img.astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_AREA).astype(np.int64)


def foveate_warp(img, gamma=2.0):
    """Diagnostic only: log-polar warp.  This changes geometry."""
    yy, xx = np.mgrid[0:OUT, 0:OUT]
    u = (xx + .5) / OUT * 2 - 1
    v = (yy + .5) / OUT * 2 - 1
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    rs = np.power(r, gamma)
    cx = cy = SRC / 2
    R = SRC / 2 - 1
    mx = (cx + rs * np.cos(theta) * R).astype(np.float32)
    my = (cy + rs * np.sin(theta) * R).astype(np.float32)
    out = cv2.remap(img.astype(np.float32), mx, my, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    out[r > 1] = 0
    return np.clip(out, 0, 9).astype(np.int64), r <= 1


def foveate(img):
    """Geometry-preserving foveation: sharp center, blurred periphery.

    No coordinate warp.  The circle is masked; local shapes stay in the same
    place; only acuity falls off with eccentricity.
    """
    base = cv2.resize(img.astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_AREA)
    small = cv2.resize(base, (OUT // 4, OUT // 4), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(small, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[0:OUT, 0:OUT]
    u = (xx + .5) / OUT * 2 - 1
    v = (yy + .5) / OUT * 2 - 1
    r = np.sqrt(u * u + v * v)
    a = np.clip((r - 0.35) / (0.95 - 0.35), 0, 1)
    a = a * a * (3 - 2 * a)  # smoothstep
    out = (1 - a[..., None]) * base.astype(np.float32) + a[..., None] * blurred.astype(np.float32)
    out[r > 1] = 0
    return np.clip(out, 0, 9).astype(np.int64), r <= 1


def central_sub(rng):
    xx = np.repeat(np.arange(OUT), OUT * 240)
    yy = np.tile(np.repeat(np.arange(OUT), 240), OUT)
    cm = (xx >= (OUT - S) // 2) & (xx < (OUT + S) // 2) & (yy >= (OUT - S) // 2) & (yy < (OUT + S) // 2)
    return np.sort(rng.choice(np.where(cm)[0], 50000, replace=False))


def main():
    rng = np.random.default_rng(0)
    patterns = make_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    sub = central_sub(rng)

    def feat(img):
        h = model.forward_numpy(encode(img[None]))[-1][0] > 0
        return h[sub]

    data = {"uniform": [], "foveated": []}
    masks = []
    examples = None
    for k, p in enumerate(patterns):
        for j in range(12):
            src = source_image(p, rng)
            u = uniform(src)
            f, mask = foveate(src)
            data["uniform"].append((feat(u), k, j < 6))
            data["foveated"].append((feat(f), k, j < 6))
            if examples is None:
                examples = (src, u, f, mask)

    out = {}
    for name, rows in data.items():
        feats = np.stack([x[0] for x in rows])
        labels = np.array([x[1] for x in rows])
        train = np.array([x[2] for x in rows])
        proto = np.stack([feats[(labels == k) & train].mean(0) for k in range(10)])
        correct = 0; gaps = []
        for k in range(10):
            for f in feats[(labels == k) & (~train)]:
                d = -np.abs(proto - f).sum(1)
                pred = int(np.argmax(d))
                correct += int(pred == k)
                gaps.append(float(d[k] - np.delete(d, k).max()))
        out[name] = {"acc": correct / 60, "gap": float(np.mean(gaps))}

    out["out_size"] = OUT
    out["geometry"] = "preserved; peripheral blur only"
    with open("results/foveated.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    src, u, f, mask = examples
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    axes[0].imshow(src / 9.0); axes[0].set_title("source 256×256", fontsize=9)
    axes[1].imshow(u / 9.0); axes[1].set_title("uniform 160×160", fontsize=9)
    axes[2].imshow(f / 9.0); axes[2].set_title("foveated: sharp center, blurred edge", fontsize=9)
    axes[3].imshow(mask, cmap="gray"); axes[3].set_title("circular field", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("圆形视野：不扭曲几何，只降低边缘清晰度", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_foveated.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_foveated.png")


if __name__ == "__main__":
    main()
