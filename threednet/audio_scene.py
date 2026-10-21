"""Auditory 3D local network: can it separate superimposed sources?

10 harmonic sources -> log-frequency spectrogram (64 x 32) -> 3-channel
(magnitude, time difference, frequency difference) -> fixed 3D local network.
Mixtures of two sources are tested at several SNRs; source prototypes come
from clean single-source examples.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.signal import stft

from .local_patterns import encode
from .model import Local3DNet

try:
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 180

FS = 16000
DUR = .5
NF, NT = 64, 32
DIMS = [(NF, NT, 60), (NF, NT, 60), (NF, NT, 60), (NF, NT, 120), (NF, NT, 240)]
N = [d[0] * d[1] * d[2] for d in DIMS]
KS = [int(n * s) for n, s in zip(N, [.5, .4, .3, .2, .1])]
K = 10


def source(k, rng):
    t = np.arange(int(FS * DUR)) / FS
    f0 = 110 + 20 * k
    wave = np.zeros_like(t)
    for h in range(1, 5):
        amp = 1.0 / h ** (1.0 + .2 * (k % 3))
        wave += amp * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
    # slow amplitude modulation, different per source
    wave *= 1 + .3 * np.sin(2 * np.pi * (2 + k % 4) * t)
    return wave / (np.max(np.abs(wave)) + 1e-9)


def add_noise(x, snr_db, rng):
    if snr_db is None:
        return x
    p = np.mean(x ** 2)
    noise = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), len(x))
    return x + noise


def spec3(wave):
    f, t, Z = stft(wave, FS, nperseg=256, noverlap=192)
    mag = np.abs(Z)
    # log-frequency interpolation
    fb = np.geomspace(80, 4000, NF)
    out = np.zeros((NF, len(t)))
    for i, ff in enumerate(fb):
        j = np.argmin(np.abs(f - ff))
        out[i] = mag[j]
    # time resample to NT
    idx = np.linspace(0, out.shape[1] - 1, NT).astype(int)
    out = out[:, idx]
    db = 20 * np.log10(out + 1e-6)
    db = np.clip((db - db.max()) / 60 + 1, 0, 1)
    q = np.round(db * 9).astype(np.int64)
    dt = np.zeros_like(q); dt[:, 1:] = np.abs(np.diff(q, axis=1))
    df = np.zeros_like(q); df[1:] = np.abs(np.diff(q, axis=0))
    return np.stack([q, np.minimum(dt, 9), np.minimum(df, 9)], -1)


def main():
    rng = np.random.default_rng(0)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)

    def feat(levels):
        return model.forward_numpy(encode(levels[None]))[-1][0] > 0

    protos = []
    for k in range(K):
        fs = [feat(spec3(source(k, rng))) for _ in range(5)]
        protos.append(np.mean(fs, 0))
    protos = np.stack(protos)
    raw_proto = np.stack([spec3(source(k, rng))[:, :, 0].reshape(-1) for k in range(K)])

    pairs = [(i, (i + 1 + rng.integers(1, 5)) % K) for i in range(100)]
    rows = {}
    for snr in [None, 20, 10, 5, 0]:
        hits = []; raw_hits = []
        for a, b in pairs:
            mix = .5 * source(a, rng) + .5 * source(b, rng)
            mix = add_noise(mix, snr, rng)
            s = spec3(mix)
            f = feat(s)
            d = -np.abs(protos - f).sum(1)
            top = set(np.argsort(-d)[:2].tolist())
            hits.append(len(top & {a, b}) / 2)
            flat = s[:, :, 0].reshape(-1).astype(np.float32)
            # raw magnitude baseline uses first channel only
            rd = -np.abs(raw_proto - flat).sum(1)
            rtop = set(np.argsort(-rd)[:2].tolist())
            raw_hits.append(len(rtop & {a, b}) / 2)
        rows[str(snr)] = {"network": float(np.mean(hits)), "raw": float(np.mean(raw_hits))}
        print(rows[str(snr)], flush=True)

    out = {"sources": K, "mixtures": 100, "snr": rows}
    with open("results/audio_scene.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["clean", "20dB", "10dB", "5dB", "0dB"]
    xs = np.arange(len(labels))
    axes[0].plot(xs, [rows[str(s)]["network"] for s in [None, 20, 10, 5, 0]], marker="o", label="network")
    axes[0].plot(xs, [rows[str(s)]["raw"] for s in [None, 20, 10, 5, 0]], marker="o", label="raw")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("source recall@2"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    axes[1].imshow(spec3(.5 * source(0, rng) + .5 * source(1, rng))[:, :, 0] / 9,
                   aspect="auto", cmap="magma")
    axes[1].set_title("mixture spectrogram (64×32)", fontsize=9)
    fig.suptitle("Auditory local network: 10 harmonic sources, 2-source mixtures", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_audio_scene.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_audio_scene.png")


if __name__ == "__main__":
    main()

