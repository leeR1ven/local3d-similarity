"""Stable-ID local edits inside the actual 3D network.

Each layer has padded slots and an alive mask.  Deletion marks dead; addition
uses new slots and locally rewires a few downstream inputs.  No np.delete and
no full-layer rebuild.
"""
from __future__ import annotations

import json

import numpy as np

from .local_patterns import encode, pattern
from .model import Local3DNet

H = W = 32
S = 16
DIMS = [(H, W, 60), (H, W, 30), (H, W, 30), (H, W, 60), (H, W, 120)]
KS = [int(H * W * 60 * .5), int(H * W * 30 * .4), int(H * W * 30 * .3),
      int(H * W * 60 * .2), int(H * W * 120 * .1)]
K = 20
FREQ = 10
PAD = 1.5


def canvas(p, rng):
    bg = rng.integers(0, 10, (8, 8, 3)).astype(np.uint8)
    bg = np.kron(bg, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.int64)
    x0 = (W - S) // 2; y0 = (H - S) // 2
    bg[y0:y0 + S, x0:x0 + S] = p
    return bg


def main():
    rng = np.random.default_rng(0)
    base = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    base.forward_numpy(np.zeros((1, DIMS[0][0] * DIMS[0][1] * DIMS[0][2]), dtype=np.float32))
    bufs = base._np_buffers
    caps = [int(d[0] * d[1] * d[2] * PAD) for d in DIMS]
    idx = []; w = []; b = []; alive = []
    for li, (ii, ww, bb) in enumerate(bufs):
        prev_n = ii.shape[0]
        cap = caps[li + 1]
        I = np.zeros((cap, 4), dtype=np.int64); Wt = np.zeros((cap, 4), np.float32)
        B = np.zeros(cap, np.float32); A = np.zeros(cap, bool)
        I[:prev_n] = ii; Wt[:prev_n] = ww; B[:prev_n] = bb; A[:prev_n] = True
        idx.append(I); w.append(Wt); b.append(B); alive.append(A)

    IDX0 = [x.copy() for x in idx]; W0 = [x.copy() for x in w]; B0 = [x.copy() for x in b]

    pats = []
    for k in range(K):
        p, m = pattern(k % 20, rng)
        p = np.resize(p, (S, S, 3)) if p.shape[0] != S else p
        full = p.copy(); full[m[:S, :S] == 0] = rng.integers(0, 10, 3)
        pats.append(full.astype(np.int64))
    images = []; labels = []; train = []
    for k, p in enumerate(pats):
        n_tr = 10 if k < FREQ else 5
        for j in range(n_tr + 5):
            images.append(canvas(p, rng)); labels.append(k); train.append(j < n_tr)
    labels = np.array(labels); train = np.array(train)
    codes = [encode(im[None]) for im in images]

    def forward():
        h = None; hs = []
        for li in range(4):
            if li == 0:
                h = codes[0] if False else None
            # input code is shared; computed per image outside
        return None

    def run_forward(code):
        h = code
        for li in range(4):
            s = (h[:, idx[li]] * w[li][None]).sum(-1) + b[li][None]
            s[:, ~alive[li]] = -np.inf
            n_alive = int(alive[li].sum())
            k = max(1, int(n_alive * KS[li + 1] / (DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2])))
            o = np.zeros_like(s)
            top = np.argpartition(-s, k - 1, axis=1)[:, :k]
            np.put_along_axis(o, top, 1.0, axis=1)
            h = o
        return h

    def features():
        F = np.zeros((len(codes), caps[-1]), dtype=bool)
        for i, c in enumerate(codes):
            F[i] = run_forward(c)[0] > 0
        return F

    def ev(F):
        proto = np.stack([F[train & (labels == k)].mean(0) for k in range(K)])
        test = ~train
        d = np.abs(F[test][:, None, :] - proto[None]).sum(2)
        pred = d.argmin(1); y = labels[test]; freq = y < FREQ
        return {"overall": float((pred == y).mean()),
                "frequent": float((pred[freq] == y[freq]).mean()),
                "rare": float((pred[~freq] == y[~freq]).mean())}

    rows = {"baseline": {"metrics": ev(features()), "alive": [int(a.sum()) for a in alive]}}
    # for each hidden layer, delete low 10% and add 5%
    for target in range(4):
        # restore arrays and alive masks
        idx = [x.copy() for x in IDX0]; w = [x.copy() for x in W0]; b = [x.copy() for x in B0]
        alive = [np.zeros(len(x), dtype=bool) for x in idx]
        for li in range(4):
            n0 = DIMS[li + 1][0] * DIMS[li + 1][1] * DIMS[li + 1][2]
            alive[li][:n0] = True
        # usage
        n0 = DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2]
        alive[target][rng.choice(n0, n0 // 10, replace=False)] = False
        # add 5% slots
        add = n0 // 20
        start = n0
        for j in range(add):
            slot = start + j
            idx[target][slot] = rng.choice(DIMS[target][0] * DIMS[target][1] * DIMS[target][2], 4)
            w[target][slot] = rng.normal(0, .1, 4); b[target][slot] = 0
            alive[target][slot] = True
            # local downstream rewire
            if target + 1 < 4:
                ids = rng.choice(DIMS[target + 1][0] * DIMS[target + 1][1] * DIMS[target + 1][2], 4, replace=False)
                idx[target + 1][ids, -1] = slot
                w[target + 1][ids, -1] = .1
        rows[f"L{target+2}"] = {"metrics": ev(features()),
                                "alive": [int(a.sum()) for a in alive]}
        print(target + 2, rows[f"L{target+2}"], flush=True)
    with open("results/stable_3d.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


