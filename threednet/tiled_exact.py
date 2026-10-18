"""Prove exact tiled equivalence: halo + global top-k.

A 108x196 canvas is split into 2x2 tiles of 54x98.  The exact tiled
implementation gathers each tile's scores from the full previous activation
(with a one-pixel halo implicitly provided by the global index) and then does
one global top-k across all tiles.  Compare with monolithic forward.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode
from .model import Local3DNet

H, W = 108, 196
TH, TW = 54, 98
DIMS = [(H, W, 60), (H, W, 15), (H, W, 15), (H, W, 15), (H, W, 15)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]


def exact_tiled(model, code):
    hs_full = model.forward_numpy(code)
    h = code.astype(np.float32)
    outs = [h]
    for l, (idx, w, b) in enumerate(model._np_buffers):
        n = idx.shape[0]
        scores = np.empty((1, n), dtype=np.float32)
        for ty in range(0, H, TH):
            for tx in range(0, W, TW):
                y1 = min(ty + TH, H); x1 = min(tx + TW, W)
                ids = []
                for y in range(ty, y1):
                    for x in range(tx, x1):
                        ids.extend(np.arange((y * W + x) * DIMS[l + 1][2],
                                             (y * W + x + 1) * DIMS[l + 1][2]))
                ids = np.asarray(ids)
                scores[0, ids] = (h[:, idx[ids]] * w[ids]).sum(-1) + b[ids]
        ix = np.argpartition(-scores, KS[l + 1], axis=1)[:, :KS[l + 1]]
        h = np.zeros_like(scores)
        np.put_along_axis(h, ix, 1.0, axis=1)
        outs.append(h)
    exact = all(np.array_equal(a, b) for a, b in zip(hs_full, outs))
    return exact, [int(a.sum()) for a in outs], [int(b.sum()) for b in hs_full]


def per_tile(model, code):
    """The current tiled_1960 behavior: top-k independently inside each tile."""
    hs_full = model.forward_numpy(code)
    outs = [code.astype(np.float32)]
    for l, (idx, w, b) in enumerate(model._np_buffers):
        scores = (outs[-1][:, idx] * w[None]).sum(-1) + b[None]
        h = np.zeros_like(scores)
        for ty in range(0, H, TH):
            for tx in range(0, W, TW):
                y1 = min(ty + TH, H); x1 = min(tx + TW, W)
                ids = []
                for y in range(ty, y1):
                    for x in range(tx, x1):
                        ids.extend(np.arange((y * W + x) * DIMS[l + 1][2],
                                             (y * W + x + 1) * DIMS[l + 1][2]))
                ids = np.asarray(ids)
                kk = int(round(len(ids) * [.5, .4, .3, .2, .1][l]))
                top = ids[np.argpartition(-scores[0, ids], kk - 1)[:kk]]
                h[0, top] = 1.0
        outs.append(h)
    return [int(a.sum()) for a in outs]


def main():
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    levels = rng.integers(0, 10, (1, H, W, 3)).astype(np.int64)
    code = encode(levels)
    exact, a, b = exact_tiled(model, code)
    out = {
        "canvas": [H, W], "tile": [TH, TW], "grid": [2, 2],
        "exact_equal": bool(exact),
        "exact_active": a, "monolithic_active": b,
        "per_tile_active": per_tile(model, code),
        "layers": len(DIMS),
    }
    with open("results/tiled_exact.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
