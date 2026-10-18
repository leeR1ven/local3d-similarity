"""Larger 196x108 LFW test with a population combination code (no unique neurons)."""
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

H, W = 196, 108
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def select_combo(all_counts, sum_counts, total_n, k, topk, central, top_similar, n_per):
    p = all_counts[k].astype(np.float32) / max(1, n_per[k])
    p_other = (sum_counts.astype(np.float32) - all_counts[k]) / max(1, total_n - n_per[k])
    p_max = np.zeros_like(p)
    for j in top_similar:
        p_max = np.maximum(p_max, all_counts[j].astype(np.float32) / max(1, n_per[j]))
    spec = p - np.maximum(p_max, p_other)
    logodds = np.log((p + 0.01) / (p_other + 0.01))
    score = spec + 0.05 * logodds
    cand = np.where(central)[0]
    top = cand[np.argsort(-score[cand])[:topk]]
    w = np.maximum(spec[top], 0.0) + 0.01
    return top, (w / w.sum()).astype(np.float32)


def main():
    d = np.load("data/lfw_large.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    n5 = N[-1]
    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        n_disc = max(1, int(round(0.6 * len(idx))))
        splits.append((idx[:n_disc], idx[n_disc:]))

    counts = np.zeros((K, n5), dtype=np.uint8)
    counts_a = np.zeros((K, n5), dtype=np.uint8)
    counts_b = np.zeros((K, n5), dtype=np.uint8)
    n_disc = np.zeros(K, np.int32); n_a = np.zeros(K, np.int32); n_b = np.zeros(K, np.int32)
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

    means = np.stack([images[splits[k][0]].mean(0).reshape(-1) for k in range(K)])
    dist = ((means[:, None] - means[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    top_similar = np.argsort(dist, axis=1)[:, :10]

    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    central = (xx >= 44) & (xx < 152) & (yy >= 0) & (yy < W)

    sum_full = counts.sum(0, dtype=np.uint32); total_full = int(n_disc.sum())
    sum_a = counts_a.sum(0, dtype=np.uint32); total_a = int(n_a.sum())
    sum_b = counts_b.sum(0, dtype=np.uint32); total_b = int(n_b.sum())
    topk = 2000
    selected, weights, selected_a, selected_b = [], [], [], []
    for k in range(K):
        s, w = select_combo(counts, sum_full, total_full, k, topk, central, top_similar[k], n_disc)
        selected.append(s); weights.append(w)
        selected_a.append(select_combo(counts_a, sum_a, total_a, k, topk, central, top_similar[k], n_a)[0])
        selected_b.append(select_combo(counts_b, sum_b, total_b, k, topk, central, top_similar[k], n_b)[0])

    stab = [len(set(selected_a[k].tolist()) & set(selected_b[k].tolist())) /
            max(len(set(selected_a[k].tolist()) | set(selected_b[k].tolist())), 1) for k in range(K)]

    correct = 0; target_rates = []; other_rates = []
    for k in range(K):
        for idx in splits[k][1]:
            levels = np.round(images[idx].astype(np.float32) / 255.0 * 9).astype(np.int64)
            h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
            scores = np.array([float((h[s] * w).sum()) for s, w in zip(selected, weights)])
            correct += int(np.argmax(scores) == k)
            target_rates.append(float((h[selected[k]] * weights[k]).sum()))
            other_rates.append(float(np.delete(scores, k).max()))
    rank1 = correct / max(1, sum(len(splits[k][1]) for k in range(K)))
    target_rates = np.array(target_rates); other_rates = np.array(other_rates)
    gap = target_rates - other_rates
    pair = np.unravel_index(np.argmin(dist), dist.shape); pair = [int(pair[0]), int(pair[1])]
    pair_overlap = len(set(selected[pair[0]].tolist()) & set(selected[pair[1]].tolist())) / topk

    out = {
        "input": "196x108 color", "n_identities": K, "n_images": int(len(images)),
        "mean_discovery": float(n_disc.mean()), "mean_validation": float(np.mean([len(splits[k][1]) for k in range(K)])),
        "rank1_sanity": float(rank1), "chance": float(1 / K),
        "target_score_mean": float(target_rates.mean()), "max_other_score_mean": float(other_rates.mean()),
        "gap_mean": float(gap.mean()), "gap_positive_fraction": float((gap > 0).mean()),
        "stability_jaccard_mean": float(np.mean(stab)), "stability_jaccard_min": float(np.min(stab)),
        "nonempty_ids": int(sum(len(s) > 0 for s in selected)),
        "combo_size": topk,
        "closest_pair": pair, "closest_pair_name": [str(names[pair[0]]), str(names[pair[1]])],
        "closest_pair_overlap": float(pair_overlap),
    }
    with open("results/lfw_large.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].hist(gap, bins=40, color="#4C78A8"); axes[0].axvline(0, color="#D62728", ls="--")
    axes[0].set_title("196×108：目标分 - 最大他人分", fontsize=9)
    axes[1].hist(stab, bins=30, color="#54A24B")
    axes[1].set_title("A/B 组合稳定性 Jaccard", fontsize=9)
    axes[2].bar(["target", "max other"], [target_rates.mean(), other_rates.mean()], color=["#4C78A8", "#F58518"])
    axes[2].set_ylim(0, 1); axes[2].set_title(f"组合得分（Rank-1 {rank1:.1%}）", fontsize=9)
    fig.suptitle(f"LFW 大模型：{K} 人 / {len(images)} 张 196×108 真实彩色中心人脸", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_lfw_large.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_lfw_large.png")


if __name__ == "__main__":
    main()
