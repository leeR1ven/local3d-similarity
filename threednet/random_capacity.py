"""Is the random model already a strong general discriminator?

100 random 96x96 images plus 100 small perturbations.  Measure exact
collisions, same-base vs different-base separation, and pixel/activation
distance correlation.  No training, no structural changes.
"""
from __future__ import annotations

import json

import numpy as np

from scipy.spatial.distance import cdist
from .local_patterns import encode
from .model import Local3DNet

H, W = 96, 96
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]


def main():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 10, (100, H, W, 3)).astype(np.int64)
    pert = base.copy()
    for i in range(100):
        # perturb a 6x6 block; same base identity, different pixels
        y = rng.integers(0, H - 6); x = rng.integers(0, W - 6)
        pert[i, y:y + 6, x:x + 6] = rng.integers(0, 10, (6, 6, 3))
    levels = np.concatenate([base, pert], axis=0)
    labels = np.tile(np.arange(100), 2)

    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    F = np.zeros((len(levels), 30000), dtype=bool)
    sub = np.sort(rng.choice(N[-1], 30000, replace=False))
    for i in range(len(levels)):
        h = model.forward_numpy(encode(levels[i][None]))[-1][0] > 0
        F[i] = h[sub]

    # pairwise Hamming
    d = F.sum(1)[:, None] + F.sum(1)[None, :] - 2 * (F.astype(np.float32) @ F.T.astype(np.float32))
    same = labels[:, None] == labels[None, :]
    tri = np.triu_indices(len(levels), 1)
    same_d = d[tri][same[tri]]; diff_d = d[tri][~same[tri]]

    # exact collisions between different bases
    eq = (F[:, None, :] == F[None, :, :]).all(2)
    collisions = int((eq & ~same).sum())

    pix = cdist(levels.reshape(len(levels), -1).astype(np.float32),
                levels.reshape(len(levels), -1).astype(np.float32), metric="cityblock")
    rho = float(np.corrcoef(pix[tri].argsort().argsort(), d[tri].argsort().argsort())[0, 1])

    out = {
        "images": len(levels), "pairs": int(len(same_d)),
        "different_base_exact_collisions": collisions,
        "same_base_distance": float(same_d.mean()),
        "different_base_distance": float(diff_d.mean()),
        "same_diff_ratio": float(same_d.mean() / diff_d.mean()),
        "min_different_base_hamming": float(diff_d.min()),
        "pixel_vs_activation_spearman": rho,
    }
    with open("results/random_capacity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

