"""Fixed model size, variable source resolution.

The same 10 local patterns are rendered at 48..256 px, then resized to the
fixed 160x160 model input.  Prototypes are learned only at 128 px; every
other resolution is a test resolution.
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .foveated import DIMS, KS, S, make_patterns, source_image
from .foveated import central_sub
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

SRC = 256
OUT = 160
RES = [48, 64, 96, 128, 192, 256]


def render(src, res):
    small = cv2.resize(src.astype(np.uint8), (res, res), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (OUT, OUT), interpolation=cv2.INTER_LINEAR).astype(np.int64)


def main():
    rng = np.random.default_rng(0)
    patterns = make_patterns(rng)
    model = Local3DNet(DIMS, KS, fan_in=4, alpha=0.01, seed=0, bias_scale=0.0)
    sub = central_sub(rng)

    def feat(img):
        h = model.forward_numpy(encode(img[None]))[-1][0] > 0
        return h[sub]

    # fixed source per pattern; train at 128 only, test all resolutions
    srcs = [source_image(p, rng) for p in patterns]
    proto = []
    for k in range(10):
        fs = []
        for _ in range(6):
            src = source_image(patterns[k], rng)
            fs.append(feat(render(src, 128)))
        proto.append(np.mean(fs, axis=0))
    proto = np.stack(proto)

    rows = []
    for res in RES:
        correct = 0; gaps = []
        for k in range(10):
            f = feat(render(srcs[k], res))
            d = -np.abs(proto - f).sum(1)
            pred = int(np.argmax(d))
            correct += int(pred == k)
            gaps.append(float(d[k] - np.delete(d, k).max()))
        rows.append({"resolution": res, "acc": correct / 10, "gap": float(np.mean(gaps))})

    out = {"model_input": OUT, "train_resolution": 128, "rows": rows}
    with open("results/resolution_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [r["resolution"] for r in rows]
    axes[0].plot(xs, [r["acc"] for r in rows], marker="o", color="#4C78A8")
    axes[0].set_ylim(0, 1); axes[0].set_xlabel("source resolution (px)")
    axes[0].set_ylabel("retrieval accuracy"); axes[0].set_title("同一模型：分辨率变化", fontsize=9)
    axes[1].plot(xs, [r["gap"] for r in rows], marker="o", color="#F58518")
    axes[1].set_xlabel("source resolution (px)"); axes[1].set_ylabel("target - max other gap")
    axes[1].set_title("分辨率 vs 分离度", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("固定 160×160 模型，训练分辨率 128，测试 48–256", fontsize=12)
    fig.tight_layout(); fig.savefig("results/fig_resolution_test.png", bbox_inches="tight"); plt.close(fig)
    print("saved results/fig_resolution_test.png")


if __name__ == "__main__":
    main()
