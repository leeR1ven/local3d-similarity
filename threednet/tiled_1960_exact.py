"""Exact 1960x1080 streaming vs monolithic, at a memory-feasible width.

Full canvas: 1080x1960.  Hidden z=4 so the monolithic model fits in RAM.
Algorithm is size-independent: halo-gather + one global top-k per layer.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet

H, W = 1080, 1960
TH, TW = 108, 196
Z = 4
DIMS = [(H, W, 60), (H, W, Z), (H, W, Z), (H, W, Z), (H, W, Z)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]


def exact_tiled(model, code):
    h = code.astype(np.float32)
    outs = [h]
    for l, (idx, w, b) in enumerate(model._np_buffers):
        scores = np.empty((1, idx.shape[0]), dtype=np.float32)
        for ty in range(0, H, TH):
            for tx in range(0, W, TW):
                y1 = min(ty + TH, H); x1 = min(tx + TW, W)
                ids = []
                zz = DIMS[l + 1][2]
                for y in range(ty, y1):
                    for x in range(tx, x1):
                        start = (y * W + x) * zz
                        ids.extend(range(start, start + zz))
                ids = np.asarray(ids)
                scores[0, ids] = (h[:, idx[ids]] * w[ids]).sum(-1) + b[ids]
        ix = np.argpartition(-scores, KS[l + 1], axis=1)[:, :KS[l + 1]]
        h = np.zeros_like(scores)
        np.put_along_axis(h, ix, 1.0, axis=1)
        outs.append(h)
    return outs


def main():
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 10, (1, H, W, 3)).astype(np.int64)
    code = encode(levels)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    mono = model.forward_numpy(code)
    tiled = exact_tiled(model, code)
    equal = all(np.array_equal(a, b) for a, b in zip(mono, tiled))
    out = {
        "canvas": [H, W], "tile": [TH, TW], "grid": [10, 10],
        "hidden_z": Z, "exact_equal": bool(equal),
        "monolithic_active": [int(x.sum()) for x in mono],
        "tiled_active": [int(x.sum()) for x in tiled],
    }
    with open("results/tiled_1960_exact.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
