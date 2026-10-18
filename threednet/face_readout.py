"""Are different faces ever exactly the same activation pattern?

Also tests whether a simple learned readout (PCA + logistic regression)
can extract identity better than the prototype rank-1.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .model import Local3DNet

H, W = 108, 196
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def encode(levels):
    t = np.arange(10)
    upper = (levels[..., None] > t).astype(np.float32)
    lower = 1.0 - upper
    return np.stack([upper, lower], -1).reshape(levels.shape[0], -1)


def main():
    d = np.load("data/lfw_large.npz", allow_pickle=True)
    images = d["images"]; labels = d["labels"]; names = d["names"]
    K = len(names)
    rng = np.random.default_rng(0)
    splits = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        rng.shuffle(idx)
        splits.append((idx[:2], idx[2:3]))
    disc_idx = np.concatenate([s[0] for s in splits])
    val_idx = np.concatenate([s[1] for s in splits])
    eval_idx = np.concatenate([disc_idx, val_idx])
    y = labels[eval_idx]
    n_disc = len(disc_idx)
    disc_y, val_y = y[:n_disc], y[n_disc:]

    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    # central face region only
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= 20) & (xx < 88) & (yy >= 40) & (yy < 156)
    sub = np.sort(rng.choice(np.where(cm)[0], 30000, replace=False))

    feats = np.zeros((len(eval_idx), len(sub)), dtype=bool)
    for j, i in enumerate(eval_idx):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
        feats[j] = h[sub]

    D = feats[:n_disc]; V = feats[n_disc:]
    # exact pattern collisions between different identities
    eq = (V[:, None, :] == D[None, :, :]).all(2)
    # mask same identity (not a collision of different faces)
    same_id = val_y[:, None] == disc_y[None, :]
    cross_eq = int((eq & ~same_id).sum())
    dist = V.astype(np.float32).sum(1)[:, None] + D.astype(np.float32).sum(1)[None, :] - 2 * (V.astype(np.float32) @ D.astype(np.float32).T)
    min_cross = float(dist[~same_id].min())
    auc = float(roc_auc_score(same_id.ravel(), -dist.ravel()))

    # 1-NN / top-5
    nn1 = float((disc_y[dist.argmin(1)] == val_y).mean())
    top5 = float(np.mean([val_y[i] in disc_y[np.argsort(dist[i])[:5]] for i in range(len(val_y))]))

    # PCA + logistic readout (trained on discovery only)
    pca = PCA(n_components=80, random_state=0)
    Xtr = pca.fit_transform(D.astype(np.float32))
    Xte = pca.transform(V.astype(np.float32))
    clf = LogisticRegression(max_iter=500, C=0.3, solver="lbfgs")
    clf.fit(Xtr, disc_y)
    logit1 = float((clf.predict(Xte) == val_y).mean())
    # nearest centroid in PCA space
    cent = np.stack([Xtr[disc_y == k].mean(0) for k in range(K)])
    pc1 = float((np.argmin(((Xte[:, None, :] - cent[None]) ** 2).sum(2), axis=1) == val_y).mean())

    out = {
        "identities": K, "images": len(images), "split": "2 discovery + 1 validation",
        "neurons": len(sub),
        "different_face_exact_collisions": cross_eq,
        "min_cross_identity_hamming": min_cross,
        "auc": auc, "nn1": nn1, "top5": top5,
        "pca_logistic_rank1": logit1, "pca_centroid_rank1": pc1,
        "prototype_rank1": 0.0,
    }
    # prototype rank-1
    P = np.stack([D[disc_y == k].mean(0) for k in range(K)]).astype(np.float32)
    pdist = V.astype(np.float32).sum(1)[:, None] + P.sum(1)[None, :] - 2 * (V.astype(np.float32) @ P.T)
    out["prototype_rank1"] = float((pdist.argmin(1) == val_y).mean())

    with open("results/face_readout.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
