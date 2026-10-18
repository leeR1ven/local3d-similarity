"""Continuous-view identity: different views, one recurrent state.

10 synthetic identities x 3 sequences x 8 frames.  Each frame changes pose
(rotation/scale), lighting (brightness/color), occlusion and background.
A recurrent state s_t = a*s_{t-1} + (1-a)*frame_t is maintained; identity is
the trajectory/attractor, never a fixed neuron subset.
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .local_patterns import H, W, background, encode, pattern
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

S = 32
N_IDS, N_SEQ, N_FRAMES = 10, 3, 8
ALPHA = 0.8
DIMS = [(H, W, 60), (H, W, 60), (H, W, 60), (H, W, 120), (H, W, 240)]
N = [H * W * 60, H * W * 60, H * W * 60, H * W * 120, H * W * 240]
KS = [int(N[0] * .5), int(N[1] * .4), int(N[2] * .3), int(N[3] * .2), int(N[4] * .1)]


def base_patterns(rng):
    out = []
    for k in range(N_IDS):
        p48, m48 = pattern(k, rng)
        p = cv2.resize(p48, (S, S), interpolation=cv2.INTER_AREA)
        mask = m48[:S, :S] > 0
        full = p.copy()
        full[~mask] = rng.integers(0, 10, 3)
        out.append(full.astype(np.uint8))
    return out


def transform(patch, rng):
    angle = float(rng.uniform(-22, 22))
    scale = float(rng.uniform(0.85, 1.15))
    M = cv2.getRotationMatrix2D((S / 2 - .5, S / 2 - .5), angle, scale)
    warped = cv2.warpAffine(patch, M, (S, S), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    col = warped.astype(np.float32) * (1 + rng.uniform(-.25, .25))
    col = col + rng.uniform(-1.5, 1.5)
    col = np.clip(col, 0, 9)
    if rng.random() < 0.5:
        x = rng.integers(0, S - 7); y = rng.integers(0, S - 7)
        col[y:y + 7, x:x + 7] = rng.integers(0, 10, (7, 7, 3))
    col = np.clip(col + rng.normal(0, .25, col.shape), 0, 9)
    return col.astype(np.int64)


def compose(patch, bg):
    img = bg.copy()
    x0, y0 = (W - S) // 2, (H - S) // 2
    img[y0:y0 + S, x0:x0 + S] = patch
    return img


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_state(seq, alpha=ALPHA):
    state = np.zeros_like(seq[0], dtype=np.float32)
    out = []
    for a in seq:
        state = alpha * state + (1 - alpha) * a.astype(np.float32)
        n = np.linalg.norm(state)
        if n > 0:
            state /= n
        out.append(state.copy())
    return np.stack(out)


def main():
    rng = np.random.default_rng(0)
    patterns = base_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    x0, y0 = (W - S) // 2, (H - S) // 2
    xx = np.repeat(np.arange(H), W * 240)
    yy = np.tile(np.repeat(np.arange(W), 240), H)
    cm = (xx >= y0) & (xx < y0 + S) & (yy >= x0) & (yy < x0 + S)
    sub = np.where(cm)[0]

    feats = np.zeros((N_IDS, N_SEQ, N_FRAMES, len(sub)), dtype=bool)
    for k in range(N_IDS):
        for s in range(N_SEQ):
            for t in range(N_FRAMES):
                q = transform(patterns[k], rng)
                img = compose(q, background(rng))
                h = model.forward_numpy(encode(img[None]))[-1][0] > 0
                feats[k, s, t] = h[sub]

    frame_cos, state_cos = [], []
    states = np.zeros_like(feats, dtype=np.float32)
    for k in range(N_IDS):
        for s in range(N_SEQ):
            st = run_state(feats[k, s])
            states[k, s] = st
            for t in range(N_FRAMES - 1):
                frame_cos.append(cos(feats[k, s, t], feats[k, s, t + 1]))
                state_cos.append(cos(st[t], st[t + 1]))

    same_state, diff_state = [], []
    same_frame, diff_frame = [], []
    for k in range(N_IDS):
        for s in range(N_SEQ):
            for s2 in range(s + 1, N_SEQ):
                same_state.append(cos(states[k, s, -1], states[k, s2, -1]))
                same_frame.append(cos(feats[k, s, -1], feats[k, s2, -1]))
        for j in range(k + 1, N_IDS):
            diff_state.append(cos(states[k, 0, -1], states[j, 0, -1]))
            diff_frame.append(cos(feats[k, 0, -1], feats[j, 0, -1]))

    # switch: concatenate identity k then identity j; compare state change at switch
    switch_cos, within_cos = [], []
    for k in range(N_IDS):
        for j in range(k + 1, N_IDS):
            seq = np.concatenate([feats[k, 0], feats[j, 0]], axis=0)
            st = run_state(seq)
            within_cos.append(cos(st[2], st[3]))
            switch_cos.append(cos(st[7], st[8]))
    switch_drop = float(np.mean(within_cos) - np.mean(switch_cos))

    # recovery after 3 blank frames (occlusion/gap)
    gap_same, gap_diff = [], []
    for k in range(N_IDS):
        seq = feats[k, 0]
        blank = np.zeros((3, len(sub)), dtype=bool)
        st = run_state(np.concatenate([seq[:4], blank, seq[4:]], axis=0))
        before, after = st[3], st[7]
        gap_same.append(cos(before, after))
        for j in range(N_IDS):
            if j != k:
                gap_diff.append(cos(before, states[j, 0, -1]))

    # proper null: shuffle final states across identities, then regroup
    flat = states[:, :, -1].reshape(-1, states.shape[-1])
    flat = flat[np.random.default_rng(7).permutation(len(flat))]
    null_same, null_diff = [], []
    for g in range(N_IDS):
        for s in range(N_SEQ):
            for s2 in range(s + 1, N_SEQ):
                null_same.append(cos(flat[g * N_SEQ + s], flat[g * N_SEQ + s2]))
        for j in range(g + 1, N_IDS):
            null_diff.append(cos(flat[g * N_SEQ], flat[j * N_SEQ]))
    null_same_diff = float(np.mean(null_same) - np.mean(null_diff))

    out = {
        "identities": N_IDS, "sequences_per_identity": N_SEQ, "frames_per_sequence": N_FRAMES,
        "frame_consecutive_cos": float(np.mean(frame_cos)),
        "state_consecutive_cos": float(np.mean(state_cos)),
        "same_state_cos": float(np.mean(same_state)),
        "different_state_cos": float(np.mean(diff_state)),
        "same_state_ratio": float(np.mean(same_state) - np.mean(diff_state)),
        "same_frame_cos": float(np.mean(same_frame)),
        "different_frame_cos": float(np.mean(diff_frame)),
        "same_frame_ratio": float(np.mean(same_frame) - np.mean(diff_frame)),
        "within_state_cos_at_switch": float(np.mean(within_cos)),
        "switch_state_cos": float(np.mean(switch_cos)),
        "switch_drop": switch_drop,
        "gap_recovery_same_cos": float(np.mean(gap_same)),
        "gap_recovery_different_cos": float(np.mean(gap_diff)),
        "gap_recovery_difference": float(np.mean(gap_same) - np.mean(gap_diff)),
        "null_same_diff": null_same_diff,
        "alpha": ALPHA,
    }
    with open("results/temporal_identity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].bar(["frame", "state"], [np.mean(frame_cos), np.mean(state_cos)],
                color=["#F58518", "#4C78A8"])
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("consecutive cosine")
    axes[0].set_title("连续帧平滑度：状态更稳", fontsize=9)
    axes[1].bar(["same id", "different id"], [np.mean(same_state), np.mean(diff_state)],
                color=["#54A24B", "#D62728"])
    axes[1].set_ylim(-1, 1); axes[1].set_ylabel("final state cosine")
    axes[1].set_title("状态空间：同人 vs 不同人", fontsize=9)
    axes[2].bar(["within", "switch"], [np.mean(within_cos), np.mean(switch_cos)],
                color=["#4C78A8", "#D62728"])
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("state cosine")
    axes[2].set_title(f"换人时状态跳变：drop={switch_drop:.3f}", fontsize=9)
    axes[3].bar(["same id", "different id"],
                [np.mean(gap_same), np.mean(gap_diff)], color=["#54A24B", "#D62728"])
    axes[3].set_ylim(0, 1); axes[3].set_ylabel("state cosine after gap")
    axes[3].set_title("遮挡 3 帧后：状态能否接回", fontsize=9)
    fig.suptitle("连续观察：不同姿势/光线 → 不同帧激活，但同一状态轨迹", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_temporal_identity.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_temporal_identity.png")


if __name__ == "__main__":
    main()
