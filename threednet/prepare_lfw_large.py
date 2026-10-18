"""Center-crop LFW faces to 196x108 color (larger model input)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/lfw/lfw")
OUT = Path("data/lfw_large.npz")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def center_face(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    side = 1.8 * max(w, h)
    x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2)); y1 = int(round(cy + side / 2))
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - img.shape[1]); pad_b = max(0, y1 - img.shape[0])
    if any((pad_l, pad_t, pad_r, pad_b)):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (196, 196), interpolation=cv2.INTER_AREA)
    crop = crop[44:152, :]  # center 196x108, face centered
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-images", type=int, default=10)
    ap.add_argument("--max-identities", type=int, default=157)
    ap.add_argument("--max-per-id", type=int, default=12)
    args = ap.parse_args()
    dirs = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg"))
            if len(imgs) >= args.min_images:
                dirs.append((d.name, imgs[: args.max_per_id]))
    dirs = dirs[: args.max_identities]
    images, labels, names, counts = [], [], [], []
    failed = 0
    for name, imgs in dirs:
        got = []
        for p in imgs:
            c = center_face(p)
            if c is None:
                failed += 1
                continue
            got.append(c)
        if len(got) < args.min_images:
            failed += len(got)
            continue
        new_id = len(names)
        images.extend(got)
        labels.extend([new_id] * len(got))
        names.append(name)
        counts.append(len(got))
    np.savez_compressed(OUT, images=np.stack(images), labels=np.array(labels, np.int32),
                        names=np.array(names), counts=np.array(counts))
    print("identities", len(names), "images", len(images), "failed", failed)
    print("saved", OUT)


if __name__ == "__main__":
    main()
