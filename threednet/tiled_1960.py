"""1960x1080 color tiled streaming test.

A 48x48 local patch is placed at the center of one 196x108 tile on a
1960x1080 canvas.  All 10x10 tiles are encoded independently; the best
prototype score gives both the pattern and the tile.  No full monolithic
network is built.
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from .local_patterns import H, W, encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

CH, CW = 1080, 1960
TH, TW = 108, 196  # 108 high x 196 wide
S = 48
DIMS = [(TH, TW, 60), (TH, TW, 60), (TH, TW, 60), (TH, TW, 120), (TH, TW, 240)]
N = [TH * TW * 60, TH * TW * 60, TH * TW * 60, TH * TW * 120, TH * TW * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def large_bg(rng):
    small = rng.integers(0, 10, (36, 65, 3)).astype(np.uint8)
    return cv2.resize(small, (CW, CH), interpolation=cv2.INTER_CUBIC).astype(np.int64)


def make_patterns(rng):
    out = []
    for k in range(10):
        p48, m48 = pattern(k, rng)
        full = p48.copy()
        full[m48 == 0] = rng.integers(0, 10, 3)
        out.append(full.astype(np.int64))
    return out


def make_canvas(patches, k, tx, ty, rng):
    canvas = large_bg(rng)
    p = patches[k]
    x = tx * TW + (TW - S) // 2
    y = ty * TH + (TH - S) // 2
    canvas[y:y + S, x:x + S] = p
    return canvas


def central_sub(rng):
    xx = np.repeat(np.arange(TH), TW * 240)
    yy = np.tile(np.repeat(np.arange(TW), 240), TH)
    cm = (xx >= (TH - S) // 2) & (xx < (TH + S) // 2) & (yy >= (TW - S) // 2) & (yy < (TW + S) // 2)
    return np.sort(rng.choice(np.where(cm)[0], 50000, replace=False))


def main():
    rng = np.random.default_rng(0)
    patches = make_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    sub = central_sub(rng)

    def feat(tile):
        h = model.forward_numpy(encode(tile[None]))[-1][0] > 0
        return h[sub]

    proto = []
    for k in range(10):
        fs = []
        for _ in range(5):
            canvas = large_bg(rng)
            p = patches[k]
            x = (TW - S) // 2; y = (TH - S) // 2
            canvas = canvas[:TH, :TW].copy()
            canvas[y:y + S, x:x + S] = p
            fs.append(feat(canvas))
        proto.append(np.mean(fs, axis=0))
    proto = np.stack(proto)

    correct_pat = 0
    correct_tile = 0
    near_tile = 0
    raw_correct = 0
    examples = []
    for k in range(10):
        tx = int(rng.integers(0, 10)); ty = int(rng.integers(0, 10))
        canvas = make_canvas(patches, k, tx, ty, rng)
        scores = np.full((10, 10, 10), -np.inf)
        for yy in range(10):
            for xx in range(10):
                tile = canvas[yy * TH:(yy + 1) * TH, xx * TW:(xx + 1) * TW]
                f = feat(tile)
                d = -np.abs(proto - f).sum(1)
                scores[:, yy, xx] = d
        best = np.unravel_index(np.argmax(scores), scores.shape)
        pk, py, px = best
        correct_pat += int(pk == k)
        correct_tile += int((py, px) == (ty, tx))
        near_tile += int(abs(py - ty) + abs(px - tx) <= 1)

        # raw pixel central crop baseline
        raw_scores = []
        for j in range(10):
            crop = canvas[ty * TH + (TH - S) // 2:ty * TH + (TH - S) // 2 + S,
                          tx * TW + (TW - S) // 2:tx * TW + (TW - S) // 2 + S]
            raw_scores.append(-np.abs(patches[j] - crop).sum())
        raw_correct += int(int(np.argmax(raw_scores)) == k)

        if len(examples) < 1:
            examples.append((canvas, k, tx, ty, pk, px, py))

    out = {
        "canvas": [CH, CW], "tile": [TH, TW], "grid": [10, 10],
        "patterns": 10,
        "pattern_accuracy": correct_pat / 10,
        "tile_hit": correct_tile / 10,
        "tile_adjacent_hit": near_tile / 10,
        "raw_pixel_pattern_accuracy": raw_correct / 10,
    }
    with open("results/tiled_1960.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    canvas, k, tx, ty, pk, px, py = examples[0]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(canvas / 9.0)
    for i in range(11):
        ax.axvline(i * TW, color="white", lw=.4, alpha=.5)
        ax.axhline(i * TH, color="white", lw=.4, alpha=.5)
    ax.add_patch(Rectangle((tx * TW, ty * TH), TW, TH, fill=False, ec="#2ECC71", lw=2))
    ax.add_patch(Rectangle((px * TW, py * TH), TW, TH, fill=False, ec="#E74C3C", lw=2, ls="--"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"1960×1080 tiled: true tile=({tx},{ty})  predicted=({px},{py})  pattern={pk}", fontsize=10)
    fig.tight_layout(); fig.savefig("results/fig_tiled_1960.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_tiled_1960.png")


if __name__ == "__main__":
    main()
