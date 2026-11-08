#!/usr/bin/env python3
"""Validate the course-layout YOLO tree before anyone spends GPU time on it.

    python scripts/check_yolo_dataset.py
    python scripts/check_yolo_dataset.py --sheet data/review/yolo_check.jpg

Default root is datasets/fruit_freshness (images/{split}, labels/{split}), matching
fruit_yolo.DATASET. Exits non-zero on hard failures.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
COLORS = [(46, 204, 113), (52, 152, 219), (241, 196, 15),
          (231, 76, 60), (155, 89, 182), (230, 126, 34)]


def read_yaml_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
    names, started = [], False
    for line in text.splitlines():
        if re.match(r"^names\s*:", line):
            started = True
            continue
        if started:
            item = re.match(r"^\s*-\s*(.+?)\s*$", line) or re.match(r"^\s*\d+\s*:\s*(.+?)\s*$", line)
            if item:
                names.append(item.group(item.lastindex).strip().strip("'\""))
            elif line.strip() and not line.startswith((" ", "\t")):
                break
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "datasets" / "fruit_freshness")
    ap.add_argument("--sheet", type=Path, default=None, help="write a proof image here")
    ap.add_argument("--sheet-n", type=int, default=6)
    args = ap.parse_args()

    yaml_path = args.root / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"missing {yaml_path} — run scripts/build_yolo_detection.py first")
    names = read_yaml_names(yaml_path)
    if not names:
        sys.exit(f"data.yaml in {args.root} has no parseable names")
    errors: list[str] = []
    warnings: list[str] = []

    hashes: dict[str, str] = {}
    cross_split = 0
    box_counts: Counter = Counter()
    img_counts: Counter = Counter()
    boxes_per_image: dict[str, list[int]] = defaultdict(list)
    samples: list[tuple[Path, Path]] = []

    for split in SPLITS:
        img_dir = args.root / "images" / split
        lbl_dir = args.root / "labels" / split
        if not img_dir.is_dir():
            (errors if split in ("train", "val") else warnings).append(f"{split}: missing images/")
            continue
        imgs = {p.stem: p for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
        img_counts[split] = len(imgs)
        for stem in imgs.keys() - lbls.keys():
            errors.append(f"{split}: image without label: {stem}")
        for stem in lbls.keys() - imgs.keys():
            errors.append(f"{split}: label without image: {stem}")
        for stem, img in imgs.items():
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                cross_split += 1
            hashes[digest] = split
            lbl = lbls.get(stem)
            if lbl is None:
                continue
            n_boxes = 0
            for n, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{stem}.txt:{n}: expected 5 cols, got {len(parts)}")
                    continue
                try:
                    cls, cx, cy, w, h = (float(v) for v in parts)
                except ValueError:
                    errors.append(f"{split}/{stem}.txt:{n}: non-numeric")
                    continue
                if cls != int(cls) or not 0 <= int(cls) < len(names):
                    errors.append(f"{split}/{stem}.txt:{n}: class {parts[0]} out of range")
                    continue
                if min(cx, cy, w, h) < 0 or max(cx, cy, w, h) > 1:
                    errors.append(f"{split}/{stem}.txt:{n}: coords not in 0-1")
                    continue
                if w <= 0 or h <= 0:
                    errors.append(f"{split}/{stem}.txt:{n}: zero-size box")
                    continue
                box_counts[names[int(cls)]] += 1
                n_boxes += 1
            boxes_per_image[split].append(n_boxes)
            samples.append((img, lbl))

    if cross_split:
        warnings.append(f"{cross_split} byte-identical images appear in more than one split")

    print(f"root: {args.root}")
    print(f"images: {dict(img_counts)}")
    print(f"boxes: {dict(box_counts)}")
    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("errors:")
        for e in errors[:40]:
            print(f"  - {e}")
        if len(errors) > 40:
            print(f"  … {len(errors) - 40} more")
        sys.exit(1)
    print("OK")

    if args.sheet and samples:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        pick = random.Random(0).sample(samples, min(args.sheet_n, len(samples)))
        tiles = []
        for img_path, lbl_path in pick:
            im = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            W, H = im.size
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = (float(v) for v in parts)
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                color = COLORS[int(cls) % len(COLORS)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tiles.append(im.resize((320, 320)))
        cols = min(3, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 320), (30, 30, 30))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 320, (i // cols) * 320))
        sheet.save(args.sheet, quality=90)
        print(f"wrote {args.sheet}")


if __name__ == "__main__":
    main()
