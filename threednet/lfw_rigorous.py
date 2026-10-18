"""Rigorous LFW neuron-discovery test.

157 identities, 4,286 real color LFW images, face-detected and center-cropped
to 96x96.  Per identity: discovery (60%) / validation (40%), and discovery is
split A/B for stability.  We evaluate the selected neuron sets directly:
target rate, max-other rate, gap, stability Jaccard, non-empty rate.  Rank-1
is reported only as a downstream sanity check, not as the discovery criterion.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def top_set(counts, k, central, denom):
    p = counts.astype(np.float32) / denom
    return p


def select_for(all_counts, sum_counts, total_n, k, topk, central, top_similar, n_per_id):
    """Select top-k central neurons for identity k using similar-id max."""
    p = all_counts[k].astype(np.float32) / max(1, n_per_id[k])
    p_other = (sum_counts.astype(np.float32) - all_counts[k]) / max(1, total_n - n_per_id[k])
    p_max = np.zeros_like(p)
    for j in top_similar:
        p_max = np.maximum(p_max, all_counts[j].astype(np.float32) / max(1, n_per_id[j]))
    spec = p - np.maximum(p_max, p_other)
    logodds = np.log((p + 0.01) / (p_other + 0.01))
    score = spec + 0.05 * logodds
    cand = np.where(central)[0]
    return cand[np.argsort(-score[cand])[:topk]]


def main():
    d = np.load("data/lfw_centered.npz", allow_pickle=True)
    images = d["images"]
    labels = d["labels"]
    names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    n5 = N[-1]

    # per-identity split, cap 20 images
    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        idx = idx[:20]
        n_disc = max(1, int(round(0.6 * len(idx))))
        splits.append((idx[:n_disc], idx[n_disc:]))

    counts = np.zeros((K, n5), dtype=np.uint8)
    counts_a = np.zeros((K, n5), dtype=np.uint8)
    counts_b = np.zeros((K, n5), dtype=np.uint8)
    n_disc = np.zeros(K, dtype=np.int32)
    n_a = np.zeros(K, dtype=np.int32)
    n_b = np.zeros(K, dtype=np.int32)

    t0 = 0
    for k in range(K):
        disc, _ = splits[k]
        n_disc[k] = len(disc)
        half = len(disc) // 2
        for j, idx in enumerate(disc):
            levels = np.round(images[idx].astype(np.float32) / 255.0 * 9).astype(np.int64)
            h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
            counts[k] += h
            if j < half:
                counts_a[k] += h; n_a[k] += 1
            else:
                counts_b[k] += h; n_b[k] += 1

    # mean face for top-10 similar identities: discovery images only (no validation leakage)
    means = np.stack([images[splits[k][0]].mean(0).reshape(-1) for k in range(K)])
    dist = ((means[:, None] - means[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    top_similar = np.argsort(dist, axis=1)[:, :10]

    # central face region
    xx = np.repeat(np.arange(96), 96 * 240)
    yy = np.tile(np.repeat(np.arange(96), 240), 96)
    central = (xx >= 8) & (xx < 88) & (yy >= 8) & (yy < 88)

    selected = []
    selected_a = []
    selected_b = []
    sum_full = counts.sum(0, dtype=np.uint32)
    sum_a = counts_a.sum(0, dtype=np.uint32)
    sum_b = counts_b.sum(0, dtype=np.uint32)
    total_full = int(n_disc.sum())
    total_a = int(n_a.sum())
    total_b = int(n_b.sum())
    for k in range(K):
        selected.append(select_for(counts, sum_full, total_full, k, 150, central, top_similar[k], n_disc))
        selected_a.append(select_for(counts_a, sum_a, total_a, k, 150, central, top_similar[k], n_a))
        selected_b.append(select_for(counts_b, sum_b, total_b, k, 150, central, top_similar[k], n_b))

    stab = []
    for k in range(K):
        a, b = set(selected_a[k].tolist()), set(selected_b[k].tolist())
        stab.append(len(a & b) / max(len(a | b), 1))

    correct = 0
    target_rates, other_rates = [], []
    for k in range(K):
        _, val = splits[k]
        for idx in val:
            levels = np.round(images[idx].astype(np.float32) / 255.0 * 9).astype(np.int64)
            h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
            scores = np.array([h[s].mean() for s in selected])
            pred = int(np.argmax(scores))
            correct += int(pred == k)
            target_rates.append(float(h[selected[k]].mean()))
            other_rates.append(float(np.delete(scores, k).max()))

    rank1 = correct / max(1, sum(len(splits[k][1]) for k in range(K)))
    target_rates = np.array(target_rates)
    other_rates = np.array(other_rates)
    gap = target_rates - other_rates

    # closest identity pair
    pair = np.unravel_index(np.argmin(dist), dist.shape)
    pair = [int(pair[0]), int(pair[1])]
    pair_overlap = len(set(selected[pair[0]]) & set(selected[pair[1]])) / 150

    out = {
        "n_identities": K,
        "n_images": int(len(images)),
        "mean_discovery": float(n_disc.mean()),
        "mean_validation": float(np.mean([len(splits[k][1]) for k in range(K)])),
        "rank1_sanity": float(rank1),
        "chance": float(1.0 / K),
        "target_rate_mean": float(target_rates.mean()),
        "max_other_rate_mean": float(other_rates.mean()),
        "gap_mean": float(gap.mean()),
        "gap_positive_fraction": float((gap > 0).mean()),
        "stability_jaccard_mean": float(np.mean(stab)),
        "stability_jaccard_min": float(np.min(stab)),
        "nonempty_ids": int(sum(len(s) > 0 for s in selected)),
        "closest_pair": pair,
        "closest_pair_name": [str(names[pair[0]]), str(names[pair[1]])],
        "closest_pair_selected_overlap": float(pair_overlap),
    }
    with open("results/lfw_rigorous.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    ax.hist(gap, bins=40, color="#4C78A8")
    ax.axvline(0, color="#D62728", ls="--")
    ax.set_title("验证集：目标率 - 最大他人率", fontsize=9)
    ax.set_xlabel("gap")
    ax = axes[1]
    ax.hist(stab, bins=30, color="#54A24B")
    ax.set_title("A/B 选神经元稳定性 Jaccard", fontsize=9)
    ax.set_xlabel("Jaccard")
    ax = axes[2]
    ax.bar(["target", "max other"], [target_rates.mean(), other_rates.mean()], color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1)
    ax.set_title(f"验证集平均激活率（Rank-1 sanity {rank1:.1%}）", fontsize=9)
    fig.suptitle(f"LFW 严格测试：{K} 人 / {len(images)} 张真实彩色中心人脸", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/fig_lfw_rigorous.png", bbox_inches="tight")
    plt.close(fig)
    print("saved results/fig_lfw_rigorous.png")


if __name__ == "__main__":
    main()
