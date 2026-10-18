"""Hebbian recurrent attractor on top of the 3D local encoder.

Frame activations are projected to 512 units.  A transition matrix W is
learned by a Hebbian outer-product rule from the first two sequences of each
identity; the third sequence is held out.  Identity is a state trajectory,
not a fixed neuron subset.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .temporal_identity import (
    DIMS, H, KS, N_FRAMES, N_IDS, N_SEQ, S, W as IW, base_patterns,
    compose, transform,
)
from .local_patterns import background, encode
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

N_STATE = 512
FAN = 64
BETA = 0.6


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    rng = np.random.default_rng(0)
    patches = base_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    x0, y0 = (IW - S) // 2, (H - S) // 2
    xx = np.repeat(np.arange(H), IW * 240)
    yy = np.tile(np.repeat(np.arange(IW), 240), H)
    cm = (xx >= y0) & (xx < y0 + S) & (yy >= x0) & (yy < x0 + S)
    sub = np.where(cm)[0]
    m = len(sub)
    pidx = rng.integers(0, m, (N_STATE, FAN))
    psign = rng.choice([-1.0, 1.0], (N_STATE, FAN))

    def project(a):
        return norm((a[pidx].astype(np.float32) * psign).sum(1))

    xs = np.zeros((N_IDS, N_SEQ, N_FRAMES, N_STATE), dtype=np.float32)
    for k in range(N_IDS):
        for s in range(N_SEQ):
            for t in range(N_FRAMES):
                q = transform(patches[k], rng)
                img = compose(q, background(rng))
                h = model.forward_numpy(encode(img[None]))[-1][0] > 0
                xs[k, s, t] = project(h[sub])

    # Hebbian transition matrix from train sequences (first two per id)
    W = np.zeros((N_STATE, N_STATE), dtype=np.float32)
    for k in range(N_IDS):
        for s in range(2):
            for t in range(1, N_FRAMES):
                W += np.outer(xs[k, s, t], xs[k, s, t - 1])
    W /= max(1e-6, np.linalg.norm(W, 2))
    W = 0.5 * (W + W.T)

    def run_att(seq):
        st = np.zeros(N_STATE, dtype=np.float32)
        out = []
        for v in seq:
            st = norm((1 - BETA) * st + BETA * np.tanh(W @ st + v))
            out.append(st.copy())
        return np.stack(out)

    def run_ema(seq):
        st = np.zeros(N_STATE, dtype=np.float32)
        out = []
        for v in seq:
            st = norm(0.8 * st + 0.2 * v)
            out.append(st.copy())
        return np.stack(out)

    def evaluate(runner):
        states = np.zeros_like(xs)
        for k in range(N_IDS):
            for s in range(N_SEQ):
                states[k, s] = runner(xs[k, s])
        frame_cos, state_cos = [], []
        for k in range(N_IDS):
            for s in range(N_SEQ):
                for t in range(N_FRAMES - 1):
                    frame_cos.append(cos(xs[k, s, t], xs[k, s, t + 1]))
                    state_cos.append(cos(states[k, s, t], states[k, s, t + 1]))
        same, diff = [], []
        for k in range(N_IDS):
            for s in range(N_SEQ):
                for s2 in range(s + 1, N_SEQ):
                    same.append(cos(states[k, s, -1], states[k, s2, -1]))
            for j in range(k + 1, N_IDS):
                diff.append(cos(states[k, 2, -1], states[j, 2, -1]))
        switch, within = [], []
        for k in range(N_IDS):
            for j in range(k + 1, N_IDS):
                seq = np.concatenate([xs[k, 2], xs[j, 2]], axis=0)
                st = runner(seq)
                within.append(cos(st[2], st[3]))
                switch.append(cos(st[7], st[8]))
        gap_same, gap_diff = [], []
        for k in range(N_IDS):
            blank = np.zeros((3, N_STATE), dtype=np.float32)
            st = runner(np.concatenate([xs[k, 0, :4], blank, xs[k, 0, 4:]], axis=0))
            before, after = st[3], st[7]
            gap_same.append(cos(before, after))
            for j in range(N_IDS):
                if j != k:
                    gap_diff.append(cos(before, states[j, 0, -1]))
        return {
            "frame_cos": float(np.mean(frame_cos)),
            "state_cos": float(np.mean(state_cos)),
            "same_cos": float(np.mean(same)),
            "diff_cos": float(np.mean(diff)),
            "same_diff": float(np.mean(same) - np.mean(diff)),
            "within_switch": float(np.mean(within)),
            "switch_cos": float(np.mean(switch)),
            "switch_drop": float(np.mean(within) - np.mean(switch)),
            "gap_same": float(np.mean(gap_same)),
            "gap_diff": float(np.mean(gap_diff)),
            "gap_diff_score": float(np.mean(gap_same) - np.mean(gap_diff)),
        }

    ema = evaluate(run_ema)
    att = evaluate(run_att)
    out = {"state_units": N_STATE, "fan_in": FAN, "beta": BETA, "ema": ema, "attractor": att}
    with open("results/recurrent_attractor.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    labels = ["EMA", "Hebbian attractor"]
    for ax, key, title in [
        (axes[0], "state_cos", "连续状态余弦"),
        (axes[1], "same_diff", "同身份 - 不同身份"),
        (axes[2], "gap_diff_score", "遮挡后同 - 不同"),
    ]:
        vals = [ema[key], att[key]]
        ax.bar(labels, vals, color=["#F58518", "#4C78A8"])
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle("连续身份：EMA vs Hebbian recurrent attractor", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_recurrent_attractor.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_recurrent_attractor.png")


if __name__ == "__main__":
    main()
