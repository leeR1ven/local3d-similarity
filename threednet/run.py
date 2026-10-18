"""Evaluate the fixed 3D local network (no training)."""
from __future__ import annotations

import json

import numpy as np
import torch

from purenet.encoding import IN_DIM, encode_numpy

from .model import Local3DNet

DIMS = ((16, 16, 60), (16, 16, 60), (16, 16, 60), (16, 16, 120), (16, 16, 240))
KS = (7680, 6144, 4608, 6144, 6144)


def pair_hamming(a):
    return (a[:, None, :] != a[None, :, :]).sum(-1)


def patch_of_index(indices, dim):
    z = dim[2]
    x = indices // (dim[1] * z)
    y = (indices // z) % dim[1]
    return (x // 4) * 4 + (y // 4)


def similarity(model, n=300, seed=0):
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 10, size=(n, 16, 16, 3))
    codes = encode_numpy(levels)
    hs = model.forward_numpy(codes)
    d_in = pair_hamming(codes)
    tri = np.triu_indices(n, 1)
    out = []
    for i, h in enumerate(hs):
        d = pair_hamming(h)[tri]
        x = d_in[tri]
        out.append({
            "layer": i + 1,
            "width": int(h.shape[1]),
            "active": int(h.sum(1)[0]),
            "pearson": float(np.corrcoef(x, d)[0, 1]),
            "spearman": float(np.corrcoef(x.argsort().argsort(), d.argsort().argsort())[0, 1]),
        })
    return out


def local_distribution(model, n=30, n_flip=200, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((16, 16, 3, 10, 2), dtype=bool)
    mask[4:8, 4:8] = True
    patch_bits = np.where(mask.reshape(-1))[0]
    local = [np.zeros(16) for _ in model.dims]
    glob = [np.zeros(16) for _ in model.dims]
    counts = {"local": np.zeros(5), "global": np.zeros(5)}
    for _ in range(n):
        levels = rng.integers(0, 10, size=(1, 16, 16, 3))
        code = encode_numpy(levels)
        cl = code.copy(); cg = code.copy()
        lx = rng.choice(patch_bits, n_flip, replace=False)
        gx = rng.choice(IN_DIM, n_flip, replace=False)
        cl[:, lx] = 1 - cl[:, lx]
        cg[:, gx] = 1 - cg[:, gx]
        b = model.forward_numpy(code)
        l = model.forward_numpy(cl)
        g = model.forward_numpy(cg)
        for i, (hb, hl, hg) in enumerate(zip(b, l, g)):
            dl = np.where((hb != hl)[0])[0]
            dg = np.where((hb != hg)[0])[0]
            p = patch_of_index(dl, model.dims[i])
            q = patch_of_index(dg, model.dims[i])
            local[i] += np.bincount(p, minlength=16)
            glob[i] += np.bincount(q, minlength=16)
            counts["local"][i] += len(dl)
            counts["global"][i] += len(dg)
    local = np.array(local) / n
    glob = np.array(glob) / n
    return {
        "local_patch": local.tolist(),
        "global_patch": glob.tolist(),
        "local_mean_changed": (counts["local"] / n).round(2).tolist(),
        "global_mean_changed": (counts["global"] / n).round(2).tolist(),
        "local_target_fraction": (local[:, 5] / local.sum(1)).round(3).tolist(),
        "global_target_fraction": (glob[:, 5] / glob.sum(1)).round(3).tolist(),
        "local_patches_changed": (local > 0.5).sum(1).tolist(),
        "global_patches_changed": (glob > 0.5).sum(1).tolist(),
    }


def main():
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0)
    print("layers", model.dims, "ks", model.ks)
    sim = similarity(model, n=250, seed=0)
    local = local_distribution(model, n=30, seed=0)
    out = {"similarity": sim, "local": local, "fan_in": 4, "alpha": 0.01, "dims": DIMS}
    with open("results/threednet_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(sim, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in local.items() if k not in ("local_patch", "global_patch")}, ensure_ascii=False, indent=2))
    print("saved results/threednet_metrics.json")


if __name__ == "__main__":
    main()
