"""Much harder task: 50 classes of 12x12 patches with heavy occlusion."""
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

H, W = 96, 96
S = 12
K = 50
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]
POP = 20000


def hard_patterns(rng):
    out = []
    for k in range(K):
        base, mask = pattern(k % 5, rng)
        p = cv2.resize(base, (S, S), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_NEAREST) > 127
        ang = rng.uniform(-12, 12); sc = rng.uniform(.9, 1.1)
        M = cv2.getRotationMatrix2D((S / 2 - .5, S / 2 - .5), ang, sc)
        p = cv2.warpAffine(p, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(m.astype(np.uint8), M, (S, S), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT) > 0
        full = p.astype(np.float32)
        full[~m] = rng.integers(0, 10, 3)
        full = np.clip(full + rng.normal(0, .8, full.shape), 0, 9)
        out.append(full.astype(np.int64))
    return out


def canvas(p, rng):
    bg = rng.integers(0, 10, (H // 4, W // 4, 3)).astype(np.uint8)
    bg = np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)
    q = p.copy()
    # heavy occlusion: 5x5 over a 12x12 patch
    x = rng.integers(0, S - 5); y = rng.integers(0, S - 5)
    q[y:y + 5, x:x + 5] = rng.integers(0, 10, (5, 5, 3))
    x0 = (W - S) // 2; y0 = (H - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = q
    return bg


def scores(l4, idx, w, b):
    return (l4[idx] * w).sum(1) + b


def topk_act(s, k):
    a = np.zeros(len(s), dtype=bool)
    a[np.argpartition(-s, k - 1)[:k]] = True
    return a


def evalv(feats, labels, train):
    proto = np.stack([feats[(labels == k) & train].mean(0) for k in range(K)])
    correct = 0; gaps = []
    for k in range(K):
        for f in feats[(labels == k) & (~train)]:
            d = -np.abs(proto - f).sum(1)
            correct += int(np.argmax(d) == k)
            gaps.append(float(d[k] - np.delete(d, k).max()))
    return correct / (K * 5), float(np.mean(gaps))


def feats(l4, idx, w, b, k):
    out = np.zeros((len(l4), len(idx)), dtype=bool)
    for i in range(len(l4)):
        out[i] = topk_act(scores(l4[i], idx, w, b), k)
    return out


def main():
    rng = np.random.default_rng(0)
    pats = hard_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    model.forward_numpy(np.zeros((1, N[0]), dtype=np.float32))
    idx0, w0, b0 = model._np_buffers[-1]
    idx0 = np.asarray(idx0); w0 = np.asarray(w0); b0 = np.asarray(b0)
    pick = np.sort(rng.choice(N[-1], POP, replace=False))
    idx = idx0[pick].copy(); w = w0[pick].copy(); b = b0[pick].copy()

    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        for j in range(10):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < 5)
    labels = np.array(labels); train = np.array(train)
    l4 = np.stack([model.forward_numpy(encode(im[None]))[-2][0] > 0 for im in images])

    base = feats(l4, idx, w, b, int(POP * .1))
    acc0, gap0 = evalv(base, labels, train)
    usage = base[train].sum(0); order = np.argsort(usage)
    low = set(order[:POP // 10].tolist()); high = order[-POP // 10:][::-1]

    def split(i, ww, bb, full=False):
        ni = []; nw = []; nb = []
        for j in i:
            if full:
                a1 = idx[j].copy(); a2 = idx[j].copy(); w1 = w[j] * .5; w2 = w[j] * .5
            else:
                a1 = np.array([idx[j, 0], idx[j, 1], idx[j, 0], idx[j, 1]])
                a2 = np.array([idx[j, 2], idx[j, 3], idx[j, 2], idx[j, 3]])
                w1 = np.array([w[j, 0], w[j, 1], 0, 0], dtype=np.float32)
                w2 = np.array([w[j, 2], w[j, 3], 0, 0], dtype=np.float32)
            ni.extend([a1, a2]); nw.extend([w1, w2]); nb.extend([np.array([b[j] / 2]), np.array([b[j] / 2])])
        return np.stack(ni), np.stack(nw), np.concatenate(nb)

    variants = {"baseline": (idx, w, b, 2000)}
    i1, w1, b1 = idx[order[POP // 10:]], w[order[POP // 10:]], b[order[POP // 10:]]
    variants["prune"] = (i1, w1, b1, int(len(i1) * .1))
    hi, hw, hb = split(high, None, None)
    i2 = np.vstack([idx, hi]); w2 = np.vstack([w, hw]); b2 = np.concatenate([b, hb])
    variants["split"] = (i2, w2, b2, int(len(i2) * .1))
    hf, hwf, hbf = split(high, None, None, full=True)
    i3 = np.vstack([idx, hf]); w3 = np.vstack([w, hwf]); b3 = np.concatenate([b, hbf])
    variants["split_full"] = (i3, w3, b3, int(len(i3) * .1))
    keep = [j for j in range(POP) if j not in low and j not in set(high.tolist())]
    i4 = np.vstack([idx[keep], hi]); w4 = np.vstack([w[keep], hw]); b4 = np.concatenate([b[keep], hb])
    variants["split_prune"] = (i4, w4, b4, int(len(i4) * .1))

    rows = {}
    for name, (ii, ww, bb, kk) in variants.items():
        f = feats(l4, ii, ww, bb, kk)
        rows[name] = {"acc": evalv(f, labels, train)[0], "gap": evalv(f, labels, train)[1], "n": len(ii)}
        print(name, rows[name], flush=True)
    out = {"classes": K, "patch": S, "baseline": rows["baseline"], "variants": rows}
    with open("results/structural_veryhard.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(rows); ax.bar(names, [rows[n]["acc"] for n in names], color="#4C78A8")
    ax.set_ylim(0, 1); ax.set_ylabel("50-class retrieval")
    ax.set_title("very hard 12×12 / heavy occlusion", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout(); fig.savefig("results/fig_structural_veryhard.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_structural_veryhard.png")


if __name__ == "__main__":
    main()
