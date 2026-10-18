"""Strict original method: neurons active for all A photos and no other person.

This is the literal "only fires for A" test, not top-k frequency scoring.
LFW 96x96, 157 identities, full L5 volume.
"""
from __future__ import annotations

import json

import numpy as np

from .model import Local3DNet

H, W = 96, 96
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def main():
    d = np.load("data/lfw_centered.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    n5 = N[-1]
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)

    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        n_disc = max(1, int(round(0.6 * len(idx))))
        splits.append((idx[:n_disc], idx[n_disc:]))

    disc = np.zeros((K, n5), dtype=np.uint8)
    val = np.zeros((K, n5), dtype=np.uint8)
    n_d = np.zeros(K, np.int32); n_v = np.zeros(K, np.int32)
    for k in range(K):
        for idx in splits[k][0]:
            levels = np.round(images[idx].astype(np.float32) / 255.0 * 9).astype(np.int64)
            h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
            disc[k] += h.astype(np.uint8); n_d[k] += 1
        for idx in splits[k][1]:
            levels = np.round(images[idx].astype(np.float32) / 255.0 * 9).astype(np.int64)
            h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
            val[k] += h.astype(np.uint8); n_v[k] += 1
        print(k, flush=True)

    total_d = disc.sum(0, dtype=np.uint16)
    total_v = val.sum(0, dtype=np.uint16)
    rows = []
    for k in range(K):
        other_d = total_d - disc[k]
        exclusive = (disc[k] == n_d[k]) & (other_d == 0)
        near = ((disc[k].astype(np.float32) / n_d[k] >= .8) &
                (other_d.astype(np.float32) / max(1, total_d.sum() - n_d[k]) <= .01))
        e = int(exclusive.sum()); ne = int(near.sum())
        if e:
            va = float(val[k][exclusive].mean()) if n_v[k] else 0.0
            ov = float((total_v[exclusive] - val[k][exclusive]).mean() / max(1, total_v.sum() - n_v[k])) if total_v.sum() > n_v[k] else 0.0
        else:
            va = ov = 0.0
        rows.append({"id": k, "name": str(names[k]), "exclusive": e, "near": ne,
                     "exclusive_val_A_rate": va, "exclusive_val_other_rate": ov,
                     "n_disc": int(n_d[k]), "n_val": int(n_v[k])})

    out = {
        "identities": K, "images": len(images),
        "strict_exclusive_total": int(sum(r["exclusive"] for r in rows)),
        "near_exclusive_total": int(sum(r["near"] for r in rows)),
        "ids_with_exclusive": int(sum(1 for r in rows if r["exclusive"] > 0)),
        "ids_with_near": int(sum(1 for r in rows if r["near"] > 0)),
        "rows": rows,
    }
    with open("results/exclusive_neurons.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print("top rows:")
    for r in sorted(rows, key=lambda x: (-x["exclusive"], -x["near"]))[:10]:
        print(r)


if __name__ == "__main__":
    main()
