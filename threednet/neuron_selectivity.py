"""Per-neuron identity selectivity across layers (LFW, no classification)."""
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
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    subsets = []
    for dim in DIMS:
        n = dim[0] * dim[1] * dim[2]
        subsets.append(np.sort(rng.choice(n, min(20000, n), replace=False)))
    counts = [np.zeros((len(np.unique(labels)), len(sub)), dtype=np.uint8) for sub in subsets]
    acts = [np.zeros((len(idx), len(sub)), dtype=bool) for sub in subsets]
    n_per = np.zeros(len(np.unique(labels)), np.int32)
    for row, i in enumerate(idx):
        k = int(labels[i])
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        hs = model.forward_numpy(encode(levels[None]))
        for li, h in enumerate(hs):
            a = h[0][subsets[li]] > 0
            acts[li][row] = a
            counts[li][k] += a.astype(np.uint8)
        n_per[k] += 1

    rows = []
    for li, c in enumerate(counts):
        p = c.astype(np.float32) / np.maximum(n_per, 1)[:, None]
        mx = p.max(0); mean = p.mean(0)
        sel = (mx - mean) / (mx + mean + 1e-6)
        # proper null: shuffle identity labels across images, rebuild counts
        null_means = []
        null_counts = []
        for _ in range(10):
            perm_labels = rng.permutation(labels[idx])
            cn = np.zeros_like(c)
            for row, kk in enumerate(perm_labels):
                cn[kk] += acts[li][row].astype(np.uint8)
            cp = cn.astype(np.float32) / np.maximum(n_per, 1)[:, None]
            mx2 = cp.max(0); mean2 = cp.mean(0)
            s2 = (mx2 - mean2) / (mx2 + mean2 + 1e-6)
            null_means.append(float(s2.mean()))
            null_counts.append(int((s2 > 0.5).sum()))
        rows.append({
            "layer": li + 1,
            "neurons": int(len(subsets[li])),
            "mean_max_rate": float(mx.mean()),
            "mean_selectivity": float(sel.mean()),
            "n_sel_0.5": int((sel > 0.5).sum()),
            "n_sel_0.8": int((sel > 0.8).sum()),
            "max_selectivity": float(sel.max()),
            "null_mean_selectivity": float(np.mean(null_means)),
            "null_n_sel_0.5": float(np.mean(null_counts)),
        })
    out = {"n_identities": int(len(np.unique(labels))), "n_images": int(len(idx)), "layers": rows}
    with open("results/neuron_selectivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    xs = [r["layer"] for r in rows]
    ax.bar(xs, [r["n_sel_0.5"] for r in rows], color="#4C78A8")
    ax.set_xlabel("层"); ax.set_ylabel("selectivity > 0.5 的神经元数")
    ax.set_title("每层：对某个身份明显更偏好的神经元", fontsize=9)
    ax = axes[1]
    ax.bar(xs, [r["mean_selectivity"] for r in rows], color="#F58518")
    ax.set_xlabel("层"); ax.set_ylabel("平均 selectivity")
    ax.set_title("每层平均单神经元身份选择性", fontsize=9)
    fig.suptitle("LFW：单个神经元对身份的区分度（不是分类准确率）", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_neuron_selectivity.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_neuron_selectivity.png")


if __name__ == "__main__":
    main()
