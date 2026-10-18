"""Do similar local patterns activate similar L5 neurons across many backgrounds?

20 synthetic local patterns, 20 training + 20 test backgrounds each, 196x108
color.  No identity labels are used for the network; the readout is nearest
prototype in activation space, and we separately report raw-pixel baseline.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

H, W, S = 196, 108, 48
X0, Y0 = (W - S) // 2, (H - S) // 2
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def pattern(kind, rng):
    yy, xx = np.mgrid[0:S, 0:S]
    cx = cy = S // 2
    m = np.zeros((S, S), np.uint8)
    if kind == 0:  # square
        m[10:38, 10:38] = 1
    elif kind == 1:  # circle
        m[(xx - cx) ** 2 + (yy - cy) ** 2 < 17 ** 2] = 1
    elif kind == 2:  # ring
        r = (xx - cx) ** 2 + (yy - cy) ** 2
        m[(r > 10 ** 2) & (r < 17 ** 2)] = 1
    elif kind == 3:  # triangle
        cv2.fillPoly(m, [np.array([[24, 7], [7, 40], [41, 40]], np.int32)], 1)
    elif kind == 4:  # cross
        m[8:40, 20:28] = 1; m[20:28, 8:40] = 1
    elif kind == 5:  # checker
        m[((xx // 8 + yy // 8) % 2) == 0] = 1
    elif kind == 6:  # vertical stripes
        m[(xx // 6) % 2 == 0] = 1
    elif kind == 7:  # horizontal stripes
        m[(yy // 6) % 2 == 0] = 1
    elif kind == 8:  # diagonal stripes
        m[((xx + yy) // 7) % 2 == 0] = 1
    elif kind == 9:  # ellipse
        m[((xx - cx) / 18) ** 2 + ((yy - cy) / 11) ** 2 < 1] = 1
    elif kind == 10:  # diamond
        m[np.abs(xx - cx) + np.abs(yy - cy) < 18] = 1
    elif kind == 11:  # two bars
        m[8:40, 12:18] = 1; m[8:40, 30:36] = 1
    elif kind == 12:  # L shape
        m[8:40, 10:16] = 1; m[34:40, 10:40] = 1
    elif kind == 13:  # half circle
        m[((xx - cx) ** 2 + (yy - cy) ** 2 < 18 ** 2) & (yy >= cy)] = 1
    elif kind == 14:  # random dots
        m[rng.random((S, S)) < 0.25] = 1
    elif kind == 15:  # concentric rings
        m[((np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) // 6) % 2) == 0] = 1
    elif kind == 16:  # grid
        m[(xx % 10 < 2) | (yy % 10 < 2)] = 1
    elif kind == 17:  # thick diagonal
        m[np.abs((xx - yy)) < 5] = 1
    elif kind == 18:  # corner
        m[8:16, 8:40] = 1; m[8:40, 8:16] = 1
    else:  # noise block
        m[rng.random((S, S)) < 0.5] = 1
    col = rng.integers(1, 10, 3)
    p = np.zeros((S, S, 3), np.uint8)
    p[m > 0] = col
    return p, m


def background(rng):
    kind = rng.integers(0, 7)
    if kind == 0:
        return np.full((H, W, 3), rng.integers(0, 10, 3), np.int64)
    if kind == 1:
        a = np.linspace(0, 1, W)[None, :, None]
        b = np.linspace(0, 1, H)[:, None, None]
        c1 = rng.integers(0, 10, 3); c2 = rng.integers(0, 10, 3)
        return np.round(c1 * (1 - a) + c2 * a + b * 0.5).astype(np.int64) % 10
    if kind == 2:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        r = r / r.max()
        c1 = rng.integers(0, 10, 3); c2 = rng.integers(0, 10, 3)
        return np.round(c1 * (1 - r[..., None]) + c2 * r[..., None]).astype(np.int64)
    if kind == 3:
        return rng.integers(0, 10, (H, W, 3))
    if kind == 4:
        a = ((np.arange(H)[:, None] // 12 + np.arange(W)[None, :] // 12) % 2) * 9
        return np.repeat(a[:, :, None], 3, axis=2).astype(np.int64)
    if kind == 5:
        a = ((np.arange(W)[None, :] // 10) % 2) * 9 * np.ones((H, 1))
        return np.repeat(a[:, :, None], 3, axis=2).astype(np.int64)
    small = rng.integers(0, 10, (8, 6, 3)).astype(np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.int64)


def compose(p, bg):
    img = bg.copy()
    mask = p.sum(2) > 0
    img[Y0:Y0 + S, X0:X0 + S][mask] = p[mask]
    return img


def main():
    rng = np.random.default_rng(0)
    patterns = []
    for k in range(20):
        p, m = pattern(k, rng)
        patterns.append((p, m))
    pixel_proto = np.stack([p.reshape(-1).astype(np.float32) for p, _ in patterns])
    mask_flat = [np.repeat(m.reshape(-1)[:, None], 3, axis=1).reshape(-1).astype(bool)
                 for _, m in patterns]
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    n5 = N[-1]
    sub_full = np.sort(rng.choice(n5, 20000, replace=False))
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    central_mask = (xx >= Y0) & (xx < Y0 + S) & (yy >= X0) & (yy < X0 + S)
    sub_center = np.sort(rng.choice(np.where(central_mask)[0], 20000, replace=False))

    train = {k: [] for k in range(20)}
    test = {k: [] for k in range(20)}
    raw_train = {k: [] for k in range(20)}
    raw_test = {k: [] for k in range(20)}
    for k in range(20):
        p, _ = patterns[k]
        for j in range(20):
            img = compose(p, background(rng))
            h = model.forward_numpy(encode(img[None]))[-1][0] > 0
            train[k].append((h[sub_full].astype(np.float32), h[sub_center].astype(np.float32)))
            raw_train[k].append(img[Y0:Y0 + S, X0:X0 + S].reshape(-1).astype(np.float32))
        for j in range(20):
            img = compose(p, background(rng))
            h = model.forward_numpy(encode(img[None]))[-1][0] > 0
            test[k].append((h[sub_full].astype(np.float32), h[sub_center].astype(np.float32)))
            raw_test[k].append(img[Y0:Y0 + S, X0:X0 + S].reshape(-1).astype(np.float32))

    def prototypes(store, idx):
        return np.stack([np.mean([x[idx] for x in store[k]], axis=0) for k in range(20)])
    proto_full = prototypes(train, 0)
    proto_center = prototypes(train, 1)

    def evaluate(proto, idx):
        correct = 0
        gaps = []
        for k in range(20):
            for x in test[k]:
                f = x[idx]
                scores = -np.abs(proto - f).sum(1)
                pred = int(np.argmax(scores))
                correct += int(pred == k)
                gaps.append(float(scores[k] - np.delete(scores, k).max()))
        return correct / 400, float(np.mean(gaps)), float(np.mean(np.array(gaps) > 0))

    acc_full, gap_full, pos_full = evaluate(proto_full, 0)
    acc_center, gap_center, pos_center = evaluate(proto_center, 1)

    # raw pixel baseline: masked template matching (ignore background inside crop)
    correct_raw = 0
    for k in range(20):
        for x in raw_test[k]:
            scores = []
            for j in range(20):
                m = mask_flat[j]
                scores.append(-np.abs(pixel_proto[j][m] - x[m]).sum())
            correct_raw += int(np.argmax(scores) == k)
    acc_raw = correct_raw / 400

    # pairwise pattern pixel distance vs activation prototype distance
    pd = np.abs(pixel_proto[:, None] - pixel_proto[None]).sum(-1)
    ad = np.abs(proto_full[:, None] - proto_full[None]).sum(-1)
    tri = np.triu_indices(20, 1)
    rho = float(np.corrcoef(pd[tri].argsort().argsort(), ad[tri].argsort().argsort())[0, 1])

    out = {
        "patterns": 20, "train_backgrounds_per_pattern": 20, "test_backgrounds_per_pattern": 20,
        "acc_full_l5": acc_full, "gap_full_l5": gap_full, "positive_gap_full": pos_full,
        "acc_central_l5": acc_center, "gap_central_l5": gap_center, "positive_gap_central": pos_center,
        "acc_raw_pixel": acc_raw,
        "pattern_pixel_vs_activation_spearman": rho,
    }
    with open("results/local_patterns.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].bar(["full L5", "central L5", "raw pixel"], [acc_full, acc_center, acc_raw],
                color=["#4C78A8", "#54A24B", "#F58518"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("retrieval accuracy")
    axes[0].set_title("同一局部画面跨背景检索", fontsize=9)
    axes[1].hexbin(pd[tri], ad[tri], gridsize=18, cmap="Blues", mincnt=1)
    axes[1].set_xlabel("pattern pixel distance"); axes[1].set_ylabel("activation prototype distance")
    axes[1].set_title(f"相似 pattern → 相似激活：ρ={rho:.3f}", fontsize=9)
    axes[2].imshow(np.concatenate([patterns[i][0] for i in range(8)], axis=1) / 9.0)
    axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].set_title("8 个局部 pattern", fontsize=9)
    fig.suptitle("20 个局部 pattern × 40 个背景：196×108，不训练", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_local_patterns.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_local_patterns.png")


if __name__ == "__main__":
    main()
