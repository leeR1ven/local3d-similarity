"""Learned pairwise readout: can a decoder extract same/different better?

Raw Hamming threshold is only one readout.  Here we project activations to
PCA space and train a logistic pair classifier on discovery pairs, then
evaluate held-out validation pairs.  This gives the model a fairer chance to
show same/different information.
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
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= 20) & (xx < 88) & (yy >= 40) & (yy < 156)
    sub = np.sort(rng.choice(np.where(cm)[0], 30000, replace=False))
    feats = np.zeros((len(eval_idx), len(sub)), dtype=bool)
    for j, i in enumerate(eval_idx):
        levels = np.round(images[i].astype(np.float32) / 255.0 * 9).astype(np.int64)
        h = model.forward_numpy(encode(levels[None]))[-1][0] > 0
        feats[j] = h[sub]

    pca = PCA(n_components=80, random_state=0)
    z = pca.fit_transform(feats[:n_disc].astype(np.float32))
    zv = pca.transform(feats[n_disc:].astype(np.float32))

    # training pairs: all same, sample different
    train = []
    for k in range(K):
        ids = np.where(disc_y == k)[0]
        train.append((ids[0], ids[1], 1))
    for _ in range(2000):
        a, b = rng.choice(n_disc, 2, replace=False)
        if disc_y[a] != disc_y[b]:
            train.append((a, b, 0))
    X = np.array([np.abs(z[a] - z[b]) for a, b, _ in train], dtype=np.float32)
    yy = np.array([t for _, _, t in train], dtype=np.int32)
    clf = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced", solver="lbfgs")
    clf.fit(X, yy)

    pairs = []
    same = val_y[:, None] == disc_y[None, :]
    for i in range(len(val_y)):
        for j in range(n_disc):
            pairs.append((zv[i], z[j], int(same[i, j])))
    Xv = np.array([np.abs(a - b) for a, b, _ in pairs], dtype=np.float32)
    yv = np.array([t for _, _, t in pairs], dtype=np.int32)
    score = clf.decision_function(Xv)
    pred = clf.predict(Xv)
    same_accept = float(pred[yv == 1].mean())
    diff_reject = float((1 - pred[yv == 0]).mean())
    tp = int(((pred == 1) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
    tn = int(((pred == 0) & (yv == 0)).sum()); fn = int(((pred == 0) & (yv == 1)).sum())
    out = {
        "identities": K, "train_pairs": len(train), "val_pairs": len(pairs),
        "val_same": int((yv == 1).sum()), "val_diff": int((yv == 0).sum()),
        "same_accept": same_accept, "diff_reject": diff_reject,
        "balanced_accuracy": (same_accept + diff_reject) / 2,
        "auc": float(roc_auc_score(yv, score)),
        "precision_same": tp / max(1, tp + fp),
        "precision_diff": tn / max(1, tn + fn),
        "recall_diff": tn / max(1, tn + fp),
    }
    with open("results/face_pair_readout.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
