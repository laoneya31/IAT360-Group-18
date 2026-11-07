#!/usr/bin/env python3
"""Build the six-class YOLO tree in the course layout (Philip).

    python scripts/build_dataset.py                        # freshness ImageFolder → data/processed/
    python scripts/build_yolo_detection.py --voc-state fresh \\
        --train-scenes 400 --val-scenes 50 --test-scenes 50
    python scripts/check_yolo_dataset.py
    # then Colab: fy.share_dataset()  OR  zip datasets/fruit_freshness for Drive

Writes the layout that fruit_yolo.DATASET / Colab expect:

  datasets/fruit_freshness/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt

Sources (missing ones are reported, not fatal):

  roboflow   data/raw/rotten-fruits — real fresh/rotten boxes
  voc        data/raw/fruit-detection — needs --voc-state fresh|rotten
  composites from data/processed (Kaggle single-fruit cut-outs)

If raw data lives outside this repo (e.g. ../data), pass --data-root ../data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
# Overridden in main() when --data-root is set.
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "datasets" / "fruit_freshness"
META = ROOT / "data" / "yolo"  # manifest + REPORT (not the image tree)

CLASSES = [
    "fresh_apple", "fresh_banana", "fresh_orange",
    "rotten_apple", "rotten_banana", "rotten_orange",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 640
MAX_SPRITE = 256  # cut-outs are pasted at <=0.32*SIZE, so caching them smaller is free
SEED = 42

FRUIT_WORDS = ("apple", "banana", "orange")


def _clear_split_dir(folder: Path) -> None:
    """Remove files in a split folder but keep .gitkeep."""
    if not folder.is_dir():
        return
    for f in folder.iterdir():
        if f.is_file() and f.name != ".gitkeep":
            f.unlink()


# --------------------------------------------------------------------------- helpers
def load_image(data: bytes | Path) -> Image.Image:
    """Open, honour EXIF orientation, and flatten to RGB (no RGBA in the training set)."""
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")


def letterbox(im: Image.Image, size: int = SIZE) -> tuple[Image.Image, float, int, int]:
    """Resize keeping aspect, pad to size x size. Returns (image, scale, pad_x, pad_y)."""
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), Image.LANCZOS), ((size - nw) // 2, (size - nh) // 2))
    return canvas, scale, (size - nw) // 2, (size - nh) // 2


def yolo_lines(boxes: list[tuple[str, float, float, float, float]]) -> str:
    """boxes are absolute pixel xyxy in a size x size image."""
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(SIZE), x2)
        y1, y2 = max(0.0, y1), min(float(SIZE), y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        out.append("%d %.6f %.6f %.6f %.6f" % (
            CLASS_ID[cls],
            (x1 + x2) / 2 / SIZE, (y1 + y2) / 2 / SIZE,
            (x2 - x1) / SIZE, (y2 - y1) / SIZE,
        ))
    return "\n".join(out) + ("\n" if out else "")


def map_name(name: str) -> str | None:
    """'Apple-Ripe' / 'rotten orange' / 'fresh_banana' -> one of CLASSES."""
    s = re.sub(r"[_\-/]+", " ", name.strip().lower())
    fruit = next((f for f in FRUIT_WORDS if f in s), None)
    if not fruit:
        return None
    if "rotten" in s or "overripe" in s or "stale" in s or "spoiled" in s:
        state = "rotten"
    elif "ripe" in s or "fresh" in s or "unripe" in s:
        state = "fresh" if "over" not in s else "rotten"
    else:
        return None
    return f"{state}_{fruit}"


def read_yaml_names(text: str) -> list[str]:
    """Pull the `names:` list out of a Roboflow data.yaml without needing pyyaml."""
    m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
    lines = text.splitlines()
    names, started = [], False
    for line in lines:
        if re.match(r"^names\s*:", line):
            started = True
            continue
        if started:
            item = re.match(r"^\s*-\s*(.+?)\s*$", line) or re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if item:
                names.append(item.group(item.lastindex).strip().strip("'\""))
            elif line.strip() and not line.startswith((" ", "\t")):
                break
    return names





# ---------------------------------------------------------------- source: roboflow dir
def collect_roboflow(folder: Path | None) -> list[dict]:
    """A YOLO export on disk (data.yaml + images/ + labels/)."""
    if not folder or not folder.exists():
        print(f"[roboflow] skipped — no export at {folder or 'data/raw/rotten-fruits'}"
              " (download from Roboflow Universe, it needs a signed-in session)")
        return []
    cfg = next((p for p in folder.rglob("data.yaml")), None)
    if not cfg:
        print(f"[roboflow] skipped — no data.yaml under {folder}")
        return []
    table = read_yaml_names(cfg.read_text(encoding="utf-8", errors="replace"))
    mapped = {i: map_name(n) for i, n in enumerate(table)}
    mapped = {i: c for i, c in mapped.items() if c}
    print(f"[roboflow] {len(table)} source classes, {len(mapped)} map onto our six")
    polygon_boxes = [0]
    rows = []
    for img in folder.rglob("*"):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        lbl = None
        for cand in (img.parent.parent / "labels" / (img.stem + ".txt"),
                     img.parent / (img.stem + ".txt"),
                     img.with_suffix(".txt")):
            if cand.exists():
                lbl = cand
                break
        if not lbl:
            continue
        kept = []
        for line in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = mapped.get(int(float(parts[0])))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            if not cls:
                continue
            if len(vals) == 4:
                kept.append((cls, *vals))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                # Roboflow's yolo26 export writes polygon contours (x1 y1 x2 y2 …) for some
                # labels. Reading the first four numbers as cx cy w h would be nonsense;
                # the honest conversion is the polygon's bounding box.
                xs = [min(max(v, 0.0), 1.0) for v in vals[0::2]]
                ys = [min(max(v, 0.0), 1.0) for v in vals[1::2]]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                if x2 - x1 > 0.002 and y2 - y1 > 0.002:
                    kept.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                    polygon_boxes[0] += 1
        if not kept:
            continue
        split = next((p for p in img.parts[::-1] if p in ("train", "valid", "val", "test")), "train")
        rows.append({"source": "roboflow", "image": img, "key": str(img),
                     "split": {"valid": "val"}.get(split, split), "boxes": kept, "kind": "file"})
    print(f"[roboflow] {len(rows)} images with at least one of our classes")
    if polygon_boxes[0]:
        print(f"[roboflow] {polygon_boxes[0]} polygon contour(s) converted to their bounding box")
    return rows


# ------------------------------------------------------------------- source: VOC XML
def _voc_objects(text: str, w: int, h: int, state: str) -> list[tuple[str, float, float, float, float]]:
    """Pascal VOC <object> blocks -> (class, cx, cy, bw, bh) normalised."""
    kept = []
    for obj in re.finditer(r"<object>(.*?)</object>", text, re.S):
        body = obj.group(1)
        nm = re.search(r"<name>\s*(.*?)\s*</name>", body, re.S)
        box = re.search(r"<xmin>\s*([\d.]+).*?<ymin>\s*([\d.]+).*?<xmax>\s*([\d.]+).*?<ymax>\s*([\d.]+)", body, re.S)
        if not nm or not box:
            continue
        fruit = next((f for f in FRUIT_WORDS if f in nm.group(1).lower()), None)
        if not fruit:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.groups())
        kept.append((f"{state}_{fruit}", (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                     (x2 - x1) / w, (y2 - y1) / h))
    return kept


def _split_from(path: str) -> str:
    token = next((p for p in path.lower().replace("\\", "/").split("/")
                  if p in ("train", "valid", "val", "test")), "train")
    return {"valid": "val"}.get(token, token)


def collect_voc(voc: Path | None, state: str | None) -> list[dict]:
    """Pascal VOC boxes, from a zip or from an extracted folder.

    That source carries no freshness label at all, so the caller must state which state
    the whole set represents — refusing by default is the point: stamping a freshness
    label onto a box without evidence would poison the class.
    """
    if not voc or not voc.exists():
        print("[voc] skipped — no archive or folder")
        return []
    if state not in ("fresh", "rotten"):
        print(f"[voc] skipped — {voc} has no freshness labels; pass --voc-state fresh|rotten")
        return []
    rows = []
    if voc.is_dir():
        for xml in sorted(voc.rglob("*.xml")):
            img = next((xml.with_suffix(e) for e in (".jpg", ".jpeg", ".png")
                        if xml.with_suffix(e).exists()), None)
            if not img:
                continue
            text = xml.read_text(encoding="utf-8", errors="replace")
            with Image.open(img) as probe:
                w, h = probe.size
            kept = _voc_objects(text, w, h, state)
            if kept:
                rows.append({"source": "voc", "image": img, "key": xml.stem,
                             "split": _split_from(str(img.relative_to(voc))), "boxes": kept,
                             "kind": "file"})
    else:
        with zipfile.ZipFile(voc) as zf:
            names = zf.namelist()
            for xml in (n for n in names if n.lower().endswith(".xml")):
                text = zf.read(xml).decode("utf-8", "replace")
                stem = Path(xml).with_suffix("").as_posix()
                img = next((n for n in (stem + e for e in (".jpg", ".jpeg", ".png"))
                            if n in names), None)
                if not img:
                    continue
                with Image.open(io.BytesIO(zf.read(img))) as probe:
                    w, h = probe.size
                kept = _voc_objects(text, w, h, state)
                if kept:
                    rows.append({"source": "voc", "image": img, "key": stem,
                                 "split": _split_from(img), "boxes": kept, "kind": "zip"})
    print(f"[voc] {len(rows)} images with boxes")
    return rows


# ------------------------------------------------------------------- source: composites
def _largest_blob(binary: np.ndarray) -> tuple[np.ndarray, bool]:
    """Biggest 4-connected region, and whether it runs into the frame edge."""
    h, w = binary.shape
    seen = np.zeros((h, w), dtype=bool)
    best_mask, best = np.zeros((h, w), dtype=bool), 0
    for sy in range(h):
        row = binary[sy]
        for sx in range(w):
            if not row[sx] or seen[sy, sx]:
                continue
            stack, pts = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) > best:
                best = len(pts)
                best_mask = np.zeros((h, w), dtype=bool)
                ys = np.fromiter((p[0] for p in pts), dtype=np.int32, count=len(pts))
                xs = np.fromiter((p[1] for p in pts), dtype=np.int32, count=len(pts))
                best_mask[ys, xs] = True
    touches = bool(best_mask[0].any() or best_mask[-1].any()
                   or best_mask[:, 0].any() or best_mask[:, -1].any())
    return best_mask, touches


def _fill_holes(blob: np.ndarray) -> np.ndarray:
    """Close interior holes (pale highlights drop out of the colour test)."""
    img = Image.fromarray((blob * 255).astype(np.uint8))
    w, h = img.size
    for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if img.getpixel(corner) == 0:
            ImageDraw.floodfill(img, corner, 128, thresh=10)
    values = np.asarray(img)
    filled = (values == 255) | (values == 0)
    # If the flood fill stopped early on a noisy background, "holes" become the whole
    # frame. Only accept hole filling that adds a plausible amount of area.
    return blob if filled.sum() > blob.sum() * 1.35 else filled


def extract_cutout(path: Path) -> tuple[Image.Image, float] | None:
    """Cut a single fruit out of its plain studio background.

    Returns (RGBA crop of the fruit, fill ratio of the mask inside its own box). Images
    whose foreground is not a compact blob are rejected — pasting a rectangle onto a
    scene would teach the detector to fire on rectangular seams.
    """
    try:
        im = load_image(path)
    except Exception:
        return None
    small = im.copy()
    small.thumbnail((120, 120), Image.BILINEAR)
    a = np.asarray(small).astype(np.int16)
    mx, mn = a.max(axis=2), a.min(axis=2)
    # A global colour-distance threshold fails on these studio shots: the background
    # gradient differs from its own border, so the whole photo reads as foreground.
    # Fruit is either colourful or dark; paper, cloth and shelves are neither.
    fg = ((mx - mn) > 40) | (mx < 90)
    fg = np.asarray(Image.fromarray((fg * 255).astype(np.uint8))
                    .filter(ImageFilter.MedianFilter(3))) > 127
    if fg.mean() < 0.10 or fg.mean() > 0.80:
        return None
    blob, touches = _largest_blob(fg)
    if touches or not blob.any():
        return None
    blob = _fill_holes(blob)
    ys, xs = np.nonzero(blob)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fill = blob.sum() / max(1, (y1 - y0) * (x1 - x0))
    # A fruit silhouette fills 0.6-0.85 of its own box. Near 1.0 means the "blob" is the
    # whole inner photo of a white-margined scan — pasting that is the rectangle bug.
    if fill < 0.45 or fill > 0.96:
        return None
    mask_img = Image.fromarray((blob * 255).astype(np.uint8)).resize(im.size, Image.NEAREST)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))  # grow 1 px, avoids a bright halo
    rgba = im.convert("RGBA")
    rgba.putalpha(mask_img)
    sx, sy = im.width / small.width, im.height / small.height
    crop = rgba.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
    if min(crop.size) < 40:
        return None
    if max(crop.size) > MAX_SPRITE:
        s = MAX_SPRITE / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    return crop, fill


def scene_background(rng: random.Random) -> Image.Image:
    """A dull tabletop: base colour, soft vertical gradient, grain, occasional edge."""
    top = np.array([rng.randint(120, 210) for _ in range(3)], dtype=np.float32)
    bottom = np.clip(top * rng.uniform(0.55, 0.85), 0, 255)
    grad = np.linspace(0, 1, SIZE, dtype=np.float32)[:, None, None]
    base = top[None, None, :] * (1 - grad) + bottom[None, None, :] * grad
    base = np.repeat(base, SIZE, axis=1)
    base += np.random.default_rng(rng.getrandbits(32)).normal(0, 6, base.shape)
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if rng.random() < 0.4:  # a shelf edge / surface seam
        y = rng.randint(int(SIZE * 0.55), int(SIZE * 0.8))
        edge = Image.new("RGB", (SIZE, 3), tuple(int(v * 0.7) for v in top))
        im.paste(edge, (0, y))
    return im.filter(ImageFilter.GaussianBlur(0.6))


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def build_composites(pools: dict[str, dict[str, list[Path]]], counts: dict[str, int],
                     rng: random.Random, staging: Path) -> list[dict]:
    """Compose crowded scenes from single-fruit photos, one split at a time."""
    cache: dict[Path, tuple[Image.Image, float] | None] = {}
    rows: list[dict] = []
    stats = Counter()

    def cutout(p: Path):
        if p not in cache:
            if len(cache) > 2000:  # cut-outs only; the canvases are staged to disk
                cache.clear()
            cache[p] = extract_cutout(p)
        return cache[p]

    for split, n_target in counts.items():
        fresh = [p for c in ("fresh_apple", "fresh_banana", "fresh_orange") for p in pools[split].get(c, [])]
        rotten = [p for c in ("rotten_apple", "rotten_banana", "rotten_orange") for p in pools[split].get(c, [])]
        if not fresh or not rotten:
            print(f"[composites] {split}: not enough classes, skipped")
            continue
        made = 0
        attempts = 0
        while made < n_target and attempts < n_target * 12:
            attempts += 1
            n_fruit = rng.randint(3, 6)
            n_rotten = rng.randint(1, max(1, n_fruit - 1))
            picks = ([rng.choice(rotten) for _ in range(n_rotten)]
                     + [rng.choice(fresh) for _ in range(n_fruit - n_rotten)])
            rng.shuffle(picks)
            canvas = scene_background(rng)
            placed: list[tuple[str, tuple[int, int, int, int]]] = []
            for path in picks:
                got = cutout(path)
                if not got:
                    continue
                crop, _fill = got
                cls = path.parent.name  # the ImageFolder class, e.g. fresh_apple
                if cls not in CLASS_ID:
                    continue
                target = rng.randint(int(SIZE * 0.16), int(SIZE * 0.32))
                scale = target / max(1, crop.width)
                sprite = crop.resize((max(2, int(crop.width * scale)),
                                      max(2, int(crop.height * scale))), Image.LANCZOS)
                sprite = sprite.rotate(rng.uniform(-14, 14), expand=True, resample=Image.BICUBIC)
                if sprite.width >= SIZE or sprite.height >= SIZE:
                    continue
                best = None
                for _ in range(10):
                    x = rng.randint(0, SIZE - sprite.width)
                    y = rng.randint(0, SIZE - sprite.height)
                    rect = (x, y, x + sprite.width, y + sprite.height)
                    worst = max((overlap(rect, r) / max(1, (r[2] - r[0]) * (r[3] - r[1]))
                                 for _, r in placed), default=0.0)
                    if worst <= 0.45:
                        best = (x, y, rect)
                        break
                    if best is None or worst < max((overlap(rect, r) / max(1, (r[2] - r[0]) * (r[3] - r[1]))
                                                    for _, r in placed), default=1.0):
                        best = (x, y, rect)
                x, y, rect = best
                canvas.paste(sprite, (x, y), sprite)
                placed.append((cls, rect))
            if len(placed) < 3:
                stats["too_few_visible"] += 1
                continue
            # a fruit hidden behind a later paste keeps only a sliver: drop that label
            kept = []
            for i, (cls, rect) in enumerate(placed):
                own = max(1, (rect[2] - rect[0]) * (rect[3] - rect[1]))
                hidden = sum(overlap(rect, r) for c, r in placed[i + 1:])
                if 1 - hidden / own >= 0.35:  # label the full extent, keep it honest
                    kept.append((cls, (rect[0] + rect[2]) / 2 / SIZE, (rect[1] + rect[3]) / 2 / SIZE,
                                 (rect[2] - rect[0]) / SIZE, (rect[3] - rect[1]) / SIZE))
            if len(kept) < 3:
                stats["too_few_visible"] += 1
                continue
            canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.85, 1.15))
            canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.85, 1.12))
            name = "c%05d_%s" % (made, hashlib.md5(repr(picks).encode()).hexdigest()[:8])
            # stage to disk: holding thousands of 640x640 canvases in RAM is what
            # silently killed the first full build
            path = staging / f"{name}.jpg"
            canvas.save(path, quality=94)
            rows.append({"source": "composite", "image": path, "key": name,
                         "split": split, "boxes": kept, "kind": "file"})
            made += 1
            if made % 100 == 0:
                print(f"[composites] {split}: {made}/{n_target}", flush=True)
        stats[f"made_{split}"] = made
    print(f"[composites] {dict(stats)}")
    return rows


# ------------------------------------------------------------------------------ build
def main() -> None:
    global RAW, PROCESSED, OUT, META

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data",
                    help="folder with raw/ and processed/ (default: ./data)")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "fruit_freshness",
                    help="course-layout YOLO tree (default: datasets/fruit_freshness)")
    ap.add_argument("--roboflow-dir", type=Path, default=None)
    ap.add_argument("--voc-zip", type=Path, default=None)
    ap.add_argument("--voc-dir", type=Path, default=None,
                    help="extracted VOC folder (images + .xml)")
    ap.add_argument("--train-scenes", type=int, default=2400)
    ap.add_argument("--val-scenes", type=int, default=320)
    ap.add_argument("--test-scenes", type=int, default=320)
    ap.add_argument("--voc-state", choices=("fresh", "rotten"), default=None)
    ap.add_argument("--test", action="store_true", help="tiny run: 30/8/8 composites")
    args = ap.parse_args()

    RAW = args.data_root / "raw"
    PROCESSED = args.data_root / "processed"
    META = args.data_root / "yolo"
    OUT = args.out
    roboflow_dir = args.roboflow_dir or (RAW / "rotten-fruits")
    voc_dir = args.voc_dir or (RAW / "fruit-detection")
    voc_zip = args.voc_zip or (RAW / "mbkinaci.zip")

    if args.test:
        args.train_scenes, args.val_scenes, args.test_scenes = 30, 8, 8

    rng = random.Random(SEED)
    manifest = PROCESSED / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"missing {manifest} — run scripts/build_dataset.py first")

    # pools per split, keyed by class, straight from the classification manifest
    pools: dict[str, dict[str, list[Path]]] = {s: defaultdict(list) for s in SPLITS}
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = PROCESSED / row["filepath"]
            if row["class"] in CLASS_ID and p.exists():
                pools[row["split"]][row["class"]].append(p)
    print("[pools] " + " · ".join(f"{s}:{sum(len(v) for v in pools[s].values())}" for s in SPLITS))

    rows = collect_roboflow(roboflow_dir)
    rows += collect_voc(voc_dir if voc_dir.exists() else voc_zip, args.voc_state)
    staging = META / "_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    rows += build_composites({s: dict(pools[s]) for s in SPLITS},
                             {"train": args.train_scenes, "val": args.val_scenes,
                              "test": args.test_scenes}, rng, staging)

    if not rows:
        sys.exit("no sources produced images — nothing to build")

    # ---------- write course layout: images/{split}, labels/{split} ----------
    for split in SPLITS:
        for sub in ("images", "labels"):
            folder = OUT / sub / split
            folder.mkdir(parents=True, exist_ok=True)
            _clear_split_dir(folder)

    seen: set[str] = set()
    out_rows, dupes = [], Counter()
    box_counts: Counter = Counter()
    per_source: Counter = Counter()
    for r in rows:
        if r["kind"] == "file":
            im = load_image(Path(r["image"]))
        else:
            im = r["image"]

        digest = hashlib.md5(np.asarray(im.resize((64, 64))).tobytes()).hexdigest()
        if digest in seen:
            dupes[r["source"]] += 1
            continue
        seen.add(digest)

        if im.size != (SIZE, SIZE):
            canvas, scale, px, py = letterbox(im)
            boxes = []
            for cls, cx, cy, w, h in r["boxes"]:
                boxes.append((cls,
                              (cx * im.width) * scale + px - (w * im.width * scale) / 2,
                              (cy * im.height) * scale + py - (h * im.height * scale) / 2,
                              (cx * im.width) * scale + px + (w * im.width * scale) / 2,
                              (cy * im.height) * scale + py + (h * im.height * scale) / 2))
        else:
            canvas, boxes = im, [(c, (cx - w / 2) * SIZE, (cy - h / 2) * SIZE,
                                 (cx + w / 2) * SIZE, (cy + h / 2) * SIZE)
                                for c, cx, cy, w, h in r["boxes"]]

        text = yolo_lines(boxes)
        if not text.strip():
            continue
        stem = hashlib.md5(("%s|%s" % (r["source"], r["key"])).encode()).hexdigest()[:16]
        split = r["split"]
        dest = OUT / "images" / split / f"{stem}.jpg"
        if r["source"] == "composite" and canvas.size == (SIZE, SIZE):
            shutil.copyfile(r["image"], dest)  # already 640 — copying beats a re-encode
        else:
            canvas.save(dest, quality=92)
        (OUT / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
        for line in text.strip().splitlines():
            box_counts[CLASSES[int(line.split()[0])]] += 1
        per_source[r["source"]] += 1
        out_rows.append([f"images/{split}/{stem}.jpg",
                         f"labels/{split}/{stem}.txt",
                         split, r["source"], len(text.strip().splitlines())])

    shutil.rmtree(staging, ignore_errors=True)

    # ---------- config + manifest + report ----------
    # Matches fruit_yolo.write_config paths so local training and Drive zip agree.
    META.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"# Built by scripts/build_yolo_detection.py — course layout for fruit_yolo\n"
        f"path: {OUT.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (META / "data.yaml").write_text(yaml_text, encoding="utf-8")

    with (META / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "label", "split", "source", "n_boxes"])
        wr.writerows(sorted(out_rows, key=lambda r: (r[2], r[3], r[0])))

    totals = Counter(r[2] for r in out_rows)
    lines = ["# YOLO detection tree (course layout)", "",
             f"Images: **{len(out_rows)}** · duplicate scenes dropped: **{sum(dupes.values())}**", "",
             f"Output: `{OUT}`", "",
             "| split | images |", "| --- | --- |"]
    lines += [f"| {s} | {totals.get(s, 0)} |" for s in SPLITS]
    lines += ["", "By source:", ""] + [f"- `{k}`: {v} images" for k, v in per_source.most_common()]
    lines += ["", "| class | boxes |", "| --- | --- |"] + [f"| {c} | {box_counts.get(c, 0)} |" for c in CLASSES]
    lines += ["", "Composites are synthetic scaffolding; team photos are the honest test.", ""]
    (META / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT} · {sum(totals.values())} images · share with fy.share_dataset() or zip this folder")


if __name__ == "__main__":
    main()
