"""Shared training and evaluation code for IAT 360 Group 18 (fruit freshness detection).

Every member's Colab notebook imports this file, so all runs use the same baseline
settings, the same metrics and the same output layout. That is what makes the
tuning results comparable. Change BASELINE only as a group decision.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config.yaml"
DATASET = REPO_ROOT / "datasets" / "fruit_freshness"
TEAM_FOLDER = "IAT360-Group 18"
DATASET_ZIP_NAME = "fruit_freshness.zip"

# Class order is fixed. It matches the alphabetical order Roboflow exports, so ids
# stay correct even if a label file came from a project spelled "rotton_*".
CLASS_NAMES = ["fresh_apple", "fresh_banana", "fresh_orange",
               "rotten_apple", "rotten_banana", "rotten_orange"]
NC = len(CLASS_NAMES)
ROTTEN_IDS = (3, 4, 5)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
WORKING_CONF = 0.25  # confidence used for accuracy and the confusion matrix

# The group baseline. These are Ultralytics defaults, with two deliberate exceptions.
#  * optimizer: the default "auto" IGNORES lr0, so lr0 tuning would change nothing.
#    "auto" resolves to AdamW, lr0 = 0.002 * 5 / (4 + nc) = 0.001, momentum 0.9 and
#    warmup_bias_lr 0 for a dataset this size. Writing those out keeps that behaviour
#    and makes lr0 a real hyperparameter.
#  * epochs / patience: the default 100 epochs is too slow for 13+ runs on free Colab.
BASELINE = dict(model="yolov8s.pt", epochs=50, patience=20, imgsz=640, batch=16,
                optimizer="AdamW", lr0=0.001, momentum=0.9, warmup_bias_lr=0.0,
                seed=0, deterministic=True)

# One hyperparameter per member, three values each, no value equal to the baseline.
TUNING_PLAN = {"Philip": ("lr0", [0.0003, 0.003, 0.01]),
               "Jasper": ("imgsz", [416, 512, 800]),
               "Daksh": ("batch", [8, 32, 64]),
               "Lavanya": ("freeze", [5, 10, 15])}

# Small, fast settings used only to check that the code runs end to end.
REHEARSAL = dict(model="yolov8n.pt", epochs=3, patience=3, imgsz=320, batch=16)


# ---------------------------------------------------------------- paths and storage
def find_team_drive():
    """Return the shared Drive folder, or None if Drive or the shortcut is missing."""
    drive = Path("/content/drive")
    candidates = [drive / "MyDrive" / TEAM_FOLDER]
    candidates += list(drive.glob(f".shortcut-targets-by-id/*/{TEAM_FOLDER}"))
    candidates += list(drive.glob(f"Shareddrives/*/{TEAM_FOLDER}"))
    return next((c for c in candidates if c.is_dir()), None)


def output_root(rehearsal=False):
    """Where runs, weights, metrics and figures are stored.

    Real runs go to the shared Drive folder so they survive a Colab disconnect and
    everyone sees the same files. Rehearsals stay on the Colab machine so they can
    never be mistaken for real results.
    """
    if rehearsal:
        return Path("/content/rehearsal") if Path("/content").exists() else REPO_ROOT / "rehearsal"
    team = find_team_drive()
    return (team / "results") if team else REPO_ROOT / "results"


def dataset_ready(root=DATASET):
    root = Path(root)
    return all(images_in(root / "images" / s) for s in ("train", "val"))


def _zip_splits(zip_path):
    """Which splits hold images in a dataset zip."""
    with zipfile.ZipFile(zip_path) as z:
        return {n.split("/")[1] for n in z.namelist()
                if n.startswith("images/") and n.count("/") == 2 and Path(n).suffix.lower() in IMG_EXTS}


def fetch_dataset(root=DATASET):
    """Make datasets/fruit_freshness match the team zip on Drive.

    A newer zip replaces the local copy, so nobody trains on old data by accident. A zip
    with no images is refused. Without Drive, a dataset already in place is used as it is.
    """
    root = Path(root)
    team = find_team_drive()
    zip_path = team / "dataset" / DATASET_ZIP_NAME if team else None
    if zip_path is None or not zip_path.exists():
        if dataset_ready(root):
            print(f"Dataset already in place: {root}")
            return True
        print(f"Dataset not found. Expected '{TEAM_FOLDER}/dataset/{DATASET_ZIP_NAME}' on Drive.")
        print("If the folder exists, add a shortcut to it in My Drive and re-run.")
        return False

    info = zip_path.stat()
    stamp = f"{info.st_size}:{int(info.st_mtime)}"
    marker = root / ".source_zip"
    if dataset_ready(root) and marker.exists() and marker.read_text() == stamp:
        print(f"Dataset up to date: {root}")
        return True
    splits = _zip_splits(zip_path)
    if not {"train", "val"} <= splits:
        print(f"{zip_path.name} on Drive has no usable images ({info.st_size:,} bytes, splits: "
              f"{sorted(splits) or 'none'}). Ask Philip to re-upload it.")
        return False
    for split in splits:
        for folder in (root / "images" / split, root / "labels" / split):
            for path in folder.glob("*") if folder.is_dir() else []:
                if path.name != ".gitkeep":
                    path.unlink()
    for cache in (root / "labels").glob("*.cache"):
        cache.unlink()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    marker.write_text(stamp)
    print(f"Extracted {zip_path.name} ({info.st_size / 1e6:.1f} MB, splits {sorted(splits)}) to {root}")
    return dataset_ready(root)


def share_dataset(root=DATASET, force=False):
    """Validate the dataset, then upload it to the team folder as the shared zip.

    Refuses an invalid or empty dataset, and refuses to replace the zip on Drive with one
    less than half its size unless force=True, so a good dataset cannot be overwritten
    by running this in a fresh session.
    """
    root = Path(root)
    assert_valid(validate_dataset(root, verbose=False))
    team = find_team_drive()
    if team is None:
        raise RuntimeError(f"Shared folder not found. Add a shortcut to '{TEAM_FOLDER}' in My Drive.")
    target = team / "dataset" / DATASET_ZIP_NAME
    target.parent.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp())
    archive = Path(shutil.make_archive(str(stage / "fruit_freshness"), "zip", root))
    new_size = archive.stat().st_size
    if target.exists() and new_size < 0.5 * target.stat().st_size and not force:
        raise RuntimeError(f"Refusing to replace the {target.stat().st_size / 1e6:.1f} MB zip on Drive "
                           f"with a {new_size / 1e6:.1f} MB one. Use force=True if this is intended.")
    shutil.copy(archive, target)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"Shared {new_size / 1e6:.1f} MB: {target}")
    return target


def write_config(root, path):
    """Write a YOLO config file for a dataset folder and return its path."""
    root, path = Path(root), Path(path)
    lines = [f"path: {root.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if images_in(root / "images" / "test"):
        lines.append("test: images/test")
    lines += ["", f"nc: {NC}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def images_in(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


# ---------------------------------------------------------------- dataset validation
def _dhash(path):
    """64-bit difference hash: equal hashes mean visually near-identical images."""
    with Image.open(path) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((9, 8)), dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1]).tobytes().hex()


def _check_label(path, nc):
    """Return (list of class ids, errors, warnings) for one YOLO label file."""
    ids, errors, warnings, seen = [], [], [], set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        where, parts = f"{path.name}:{n}", line.split()
        if len(parts) != 5:
            hint = " (polygon label: export as bounding boxes)" if len(parts) > 5 else ""
            errors.append(f"{where} has {len(parts)} columns, expected 5{hint}")
            continue
        try:
            cls_f, x, y, w, h = (float(v) for v in parts)
        except ValueError:
            errors.append(f"{where} has a non-numeric value: {line!r}")
            continue
        if cls_f != int(cls_f) or not 0 <= cls_f < nc:
            errors.append(f"{where} class id {parts[0]} is outside 0-{nc - 1}")
            continue
        if min(x, y, w, h) < 0 or max(x, y, w, h) > 1:
            errors.append(f"{where} coordinates are not normalised to 0-1 (pixels?): {line!r}")
            continue
        if w == 0 or h == 0:
            errors.append(f"{where} has a zero-size box")
            continue
        if x - w / 2 < -1e-3 or x + w / 2 > 1 + 1e-3 or y - h / 2 < -1e-3 or y + h / 2 > 1 + 1e-3:
            warnings.append(f"{where} box extends past the image edge")
        if line in seen:
            warnings.append(f"{where} is a duplicate box")
        seen.add(line)
        ids.append(int(cls_f))
    return ids, errors, warnings


def validate_dataset(root=DATASET, nc=NC, near_duplicates=True, splits=SPLITS, verbose=True):
    """Check a YOLO dataset for the mistakes that silently damage training.

    Errors must be fixed before training. Warnings need a human decision.
    Returns a dict with errors, warnings, image counts and box counts per class.
    """
    root = Path(root)
    errors, warnings = [], []
    images = {s: 0 for s in splits}
    boxes = {s: Counter() for s in splits}
    exact, near = defaultdict(list), defaultdict(list)
    exif_rotated = 0

    for split in splits:
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        imgs = images_in(img_dir)
        images[split] = len(imgs)
        if not imgs:
            (errors if split in ("train", "val") else warnings).append(f"{split}: no images found")
            continue
        stems = {p.stem for p in imgs}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
        if len(stems) != len(imgs):
            errors.append(f"{split}: two images share a file name but differ in extension")
        for stem in sorted(set(labels) - stems):
            errors.append(f"{split}: label {stem}.txt has no matching image")
        for img in imgs:
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    if im.getexif().get(274, 1) != 1:
                        exif_rotated += 1
            except Exception as e:
                errors.append(f"{split}: {img.name} is unreadable ({type(e).__name__})")
                continue
            exact[hashlib.md5(img.read_bytes()).hexdigest()].append(f"{split}/{img.name}")
            if near_duplicates:
                near[_dhash(img)].append(f"{split}/{img.name}")
            if img.stem not in labels:
                errors.append(f"{split}: {img.name} has no label file (use an empty .txt for no fruit)")
                continue
            ids, e, w = _check_label(labels[img.stem], nc)
            errors += [f"{split}: {m}" for m in e]
            warnings += [f"{split}: {m}" for m in w]
            boxes[split].update(ids)

    for files in exact.values():
        if len(files) > 1:
            where = {f.split("/")[0] for f in files}
            message = f"identical images {files[:3]}"
            (errors if len(where) > 1 else warnings).append(
                ("LEAKAGE across splits: " if len(where) > 1 else "") + message)
    exact_sets = {frozenset(v) for v in exact.values()}
    for files in near.values():
        if len({f.split('/')[0] for f in files}) > 1 and frozenset(files) not in exact_sets:
            warnings.append(f"possible leakage, near-identical images across splits: {files[:3]}")
    if exif_rotated:
        warnings.append(f"{exif_rotated} images rely on EXIF rotation: bake it in before annotating")

    report = dict(errors=errors, warnings=warnings, images=images,
                  boxes={s: {CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i): c
                             for i, c in sorted(boxes[s].items())} for s in splits})
    if verbose:
        _print_validation(report)
    return report


def _print_validation(report):
    print(f"{'class':<16}" + "".join(f"{s:>8}" for s in report["boxes"]))
    for name in CLASS_NAMES:
        print(f"{name:<16}" + "".join(f"{report['boxes'][s].get(name, 0):>8}" for s in report["boxes"]))
    print(f"{'images':<16}" + "".join(f"{report['images'][s]:>8}" for s in report["images"]))
    for kind in ("errors", "warnings"):
        items = report[kind]
        print(f"\n{len(items)} {kind}" + (":" if items else ""))
        for m in items[:15]:
            print("  -", m)
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more")


def assert_valid(report):
    if report["errors"]:
        raise ValueError(f"Fix {len(report['errors'])} dataset errors before training")
    print("Dataset passed: safe to train.")


def check_config(path=CONFIG):
    """Confirm a config or Roboflow data.yaml uses our six classes in our order."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    fixed = [n.strip().lower().replace(" ", "_").replace("rotton", "rotten") for n in names]
    if fixed != CLASS_NAMES:
        raise ValueError(f"Class order mismatch.\n  file:     {names}\n  expected: {CLASS_NAMES}")
    if names != CLASS_NAMES:
        print(f"Note: spelling differs ({names[3]}) but class ids line up.")
    print(f"{Path(path).name}: {len(names)} classes, order matches.")
    return True


# ---------------------------------------------------------------- composite scenes
def fruit_cutout(path, white=225):
    """Cut a fruit off its plain light background. Returns RGBA, or None if it fails."""
    import cv2
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    mask = (rgb.min(axis=2) < white).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    count, comp, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[biggest]
    # Reject cut-outs that fill the frame (background was not plain) or are tiny.
    if area < 0.05 * mask.size or (w * h) > 0.97 * mask.size:
        return None
    alpha = np.where(comp == biggest, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])[y:y + h, x:x + w]
    return Image.fromarray(rgba, "RGBA")


def _background(size, rng):
    """A plain textured surface with a soft gradient, like a table or bin liner."""
    base = np.array([rng.randint(60, 220) for _ in range(3)], dtype=np.float32)
    ramp = np.linspace(-25, 25, size, dtype=np.float32)[:, None, None]
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 6, (size, size, 1))
    return Image.fromarray(np.clip(base + ramp + noise, 0, 255).astype(np.uint8), "RGB")


def build_composites(sources, out_root, counts, size=640, per_image=(2, 5), seed=0):
    """Paste single fruits onto backgrounds to make labelled multi-fruit scenes.

    sources: {class_id: [image paths]}.  counts: {split: number of scenes}.
    Source photos are divided between splits first, so one photo can never appear in
    two splits. Returns the number of boxes written per split.
    """
    rng = random.Random(seed)
    out_root = Path(out_root)
    pools = {s: {} for s in counts}
    total = sum(counts.values())
    for cid, files in sources.items():
        files = sorted(files)
        rng.shuffle(files)
        start = 0
        for s, n in counts.items():
            take = max(1, round(len(files) * n / total))
            pools[s][cid] = files[start:start + take]
            start += take

    written = {}
    for split, n_scenes in counts.items():
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        written[split] = 0
        for i in range(n_scenes):
            canvas, rows, placed = _background(size, rng), [], []
            for _ in range(rng.randint(*per_image)):
                cid = rng.choice(sorted(pools[split]))
                fruit = None
                for _try in range(5):
                    fruit = fruit_cutout(rng.choice(pools[split][cid]))
                    if fruit is not None:
                        break
                if fruit is None:
                    continue
                scale = rng.uniform(0.22, 0.42) * size / max(fruit.size)
                fruit = fruit.resize((max(1, int(fruit.width * scale)),
                                      max(1, int(fruit.height * scale))), Image.BILINEAR)
                fruit = fruit.rotate(rng.uniform(0, 360), expand=True, resample=Image.BILINEAR)
                fruit = fruit.crop(fruit.getbbox())
                if fruit.width >= size or fruit.height >= size:
                    continue
                for _try in range(20):
                    x, y = rng.randint(0, size - fruit.width), rng.randint(0, size - fruit.height)
                    box = (x, y, x + fruit.width, y + fruit.height)
                    if all(_iou(box, other) < 0.1 for other in placed):
                        canvas.paste(fruit, (x, y), fruit)
                        placed.append(box)
                        rows.append(f"{cid} {(x + fruit.width / 2) / size:.6f} "
                                    f"{(y + fruit.height / 2) / size:.6f} "
                                    f"{fruit.width / size:.6f} {fruit.height / size:.6f}")
                        break
            stem = f"composite_{split}_{i:05d}"
            canvas.save(out_root / "images" / split / f"{stem}.jpg", quality=92)
            (out_root / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            written[split] += len(rows)
    return written


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- training
def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > T4 GPU, then re-run.")
    print("GPU:", torch.cuda.get_device_name(0))


def run_settings(overrides=None, rehearsal=False):
    """The shared baseline, plus any overrides. Pass the result to model.train()."""
    return {**BASELINE, **(REHEARSAL if rehearsal else {}), **(overrides or {})}


def runs_dir(rehearsal=False):
    return output_root(rehearsal) / "runs"


def weights_path(name, rehearsal=False):
    return output_root(rehearsal) / "weights" / f"{name}.pt"


def record_run(name, owner, model, overrides=None, rehearsal=False):
    """Save best.pt, the run's settings and its plots after model.train() finishes."""
    root = output_root(rehearsal)
    save_dir = Path(model.trainer.save_dir)
    weights = weights_path(name, rehearsal)
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(save_dir / "weights" / "best.pt", weights)

    batch = int(model.trainer.batch_size)
    accumulate = max(round(int(model.trainer.args.nbs) / batch), 1)
    meta = dict(name=name, owner=owner, overrides=overrides or {}, rehearsal=rehearsal,
                minutes=_train_minutes(save_dir), batch=batch, accumulate=accumulate,
                effective_batch=batch * accumulate,
                settings={k: str(v) for k, v in vars(model.trainer.args).items()})
    _write_json(root / "metrics" / f"{name}__train.json", meta)
    print(f"{name}: finished in {meta['minutes']} min. batch {batch} x accumulate {accumulate} "
          f"= effective batch {meta['effective_batch']}. Saved {weights}")
    export_figures(name, save_dir, rehearsal)
    return weights


def _train_minutes(save_dir):
    import pandas as pd
    df = pd.read_csv(Path(save_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return round(float(df["time"].iloc[-1]) / 60, 1) if "time" in df else None


def train_run(name, owner, overrides=None, data=CONFIG, rehearsal=False, force=False, resume=False):
    """Train one model from the shared baseline, changing only `overrides`.

    A run that already finished is skipped unless force=True, so re-running a notebook
    never wastes GPU time. After a Colab disconnect, resume=True continues from last.pt.
    Returns the path to the saved weights.
    """
    from ultralytics import YOLO
    if not rehearsal:
        require_gpu()
        check_config(data)
    weights = weights_path(name, rehearsal)
    if weights.exists() and not (force or resume):
        print(f"{name}: already trained, using {weights}")
        return weights
    if resume:
        model = YOLO(str(runs_dir(rehearsal) / name / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        args = run_settings(overrides, rehearsal)
        model = YOLO(args.pop("model"))
        model.train(data=str(data), project=str(runs_dir(rehearsal)), name=name, exist_ok=True, **args)
    return record_run(name, owner, model, overrides, rehearsal)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def metrics_from_matrix(matrix, nc=NC):
    """Counts from an Ultralytics confusion matrix (rows predicted, columns true).

    The last row and column are 'background'. Row background = missed fruit, column
    background = false alarm. Detection has no true negatives, so accuracy here is
    correct boxes / (correct + wrong class + missed + false alarms).
    """
    m = np.asarray(matrix, dtype=float)
    correct = float(np.trace(m[:nc, :nc]))
    total = float(m.sum())
    rotten = list(ROTTEN_IDS)
    true_rotten = float(m[:, rotten].sum())
    flagged = float(m[np.ix_(rotten, rotten)].sum())
    fresh = [i for i in range(nc) if i not in ROTTEN_IDS]
    return dict(accuracy=correct / total if total else 0.0,
                rotten_flag_recall=flagged / true_rotten if true_rotten else 0.0,
                rotten_called_fresh=float(m[np.ix_(fresh, rotten)].sum()),
                rotten_missed=float(m[nc, rotten].sum()),
                false_alarms=float(m[:nc, nc].sum()))


def trained_imgsz(name, rehearsal=False):
    """The image size a run was trained at, so it is evaluated at the same size."""
    path = output_root(rehearsal) / "metrics" / f"{name}__train.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["settings"]["imgsz"])
    return run_settings(rehearsal=rehearsal)["imgsz"]


def evaluate(name, weights, data=CONFIG, split="val", imgsz=None, rehearsal=False):
    """Score a trained model and save every metric the rubric asks for.

    Two passes, because they answer different questions. mAP needs YOLO's default
    near-zero confidence so it can sweep the whole precision-recall curve. Accuracy and
    the confusion matrix need the confidence a real deployment would use: at the
    default, nearly every low-score box counts as a false alarm.
    """
    from ultralytics import YOLO
    root = output_root(rehearsal)
    imgsz = imgsz or trained_imgsz(name, rehearsal)
    common = dict(data=str(data), split=split, imgsz=imgsz, plots=True, exist_ok=True,
                  project=str(root / "evals"), verbose=False)
    model = YOLO(str(weights))
    result = model.val(name=f"{name}_{split}", **common)
    working = model.val(name=f"{name}_{split}_working", conf=WORKING_CONF, **common)
    for plot in Path(working.save_dir).glob("confusion_matrix*.png"):
        shutil.copy2(plot, Path(result.save_dir) / plot.name)

    box = result.box
    rows = {}
    for i, cid in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows[CLASS_NAMES[int(cid)]] = dict(precision=float(p), recall=float(r), f1=float(_f1(p, r)),
                                           mAP50=float(ap50), mAP50_95=float(ap))
    rotten = [rows[CLASS_NAMES[i]]["recall"] for i in ROTTEN_IDS if CLASS_NAMES[i] in rows]
    matrix = working.confusion_matrix.matrix
    true_boxes = int(np.asarray(matrix)[:, :NC].sum())
    expected = int(result.nt_per_class.sum())
    if true_boxes != expected:
        raise RuntimeError(f"Confusion matrix holds {true_boxes} true boxes, expected {expected}")
    out = dict(name=name, split=split, conf=WORKING_CONF, true_boxes=true_boxes,
               precision=float(box.mp), recall=float(box.mr), f1=float(_f1(box.mp, box.mr)),
               mAP50=float(box.map50), mAP50_95=float(box.map),
               rotten_recall=float(np.mean(rotten)) if rotten else 0.0, per_class=rows,
               **metrics_from_matrix(matrix))
    _write_json(root / "metrics" / f"{name}__{split}.json", out)
    export_figures(f"{name}_{split}", result.save_dir, rehearsal)
    return out


def export_figures(name, source_dir, rehearsal=False):
    """Copy YOLO's plots into one folder per run, so slides use the same figures."""
    dest = output_root(rehearsal) / "figures" / name
    dest.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir)
    kept = [p for p in source_dir.glob("*.png")
            if not p.name.startswith(("train_batch", "val_batch", "labels"))]
    for extra in ("results.csv", "args.yaml"):
        if (source_dir / extra).exists():
            kept.append(source_dir / extra)
    for p in kept:
        shutil.copy2(p, dest / p.name)
    return dest


def package_results(names, dest=None):
    """Zip metrics, figures and weights in the repo's layout, ready to commit to GitHub."""
    root = output_root()
    stage = Path("/tmp/group18_results") if Path("/tmp").exists() else REPO_ROOT / "_package"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "results" / "metrics").mkdir(parents=True)
    (stage / "results" / "figures").mkdir(parents=True)
    (stage / "weights").mkdir()
    for name in names:
        for path in (root / "metrics").glob(f"{name}__*.json"):
            shutil.copy2(path, stage / "results" / "metrics")
        for folder in (root / "figures").glob(f"{name}*"):
            if folder.is_dir():
                shutil.copytree(folder, stage / "results" / "figures" / folder.name, dirs_exist_ok=True)
        if (root / "weights" / f"{name}.pt").exists():
            shutil.copy2(root / "weights" / f"{name}.pt", stage / "weights")
    for chart in (root / "figures").glob("*.png"):
        shutil.copy2(chart, stage / "results" / "figures")
    archive = shutil.make_archive(str(Path(dest or root) / "results_for_github"), "zip", stage)
    print(f"Unzip at the top of the repo, then commit: {archive}")
    return archive


# ---------------------------------------------------------------- reporting
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "mAP50", "mAP50_95",
                  "rotten_recall", "rotten_flag_recall"]


def compare_runs(split="val", names=None, rehearsal=False, baseline="baseline"):
    """Table of every run's metrics, with each run's mAP change against the baseline."""
    import pandas as pd
    folder = output_root(rehearsal) / "metrics"
    rows = []
    for path in sorted(folder.glob(f"*__{split}.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if names and m["name"] not in names:
            continue
        train_path = folder / f"{m['name']}__train.json"
        t = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else {}
        rows.append(dict(run=m["name"], owner=t.get("owner", ""),
                         changed=", ".join(f"{k}={v}" for k, v in t.get("overrides", {}).items()) or "none",
                         **{c: m[c] for c in METRIC_COLUMNS}, minutes=t.get("minutes"),
                         effective_batch=t.get("effective_batch")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("run")
    if baseline in df.index:
        df["mAP50_95_vs_baseline"] = df["mAP50_95"] - df.loc[baseline, "mAP50_95"]
    return df.round(4)


def planned_runs():
    """Every run the group plans, with its owner, in presentation order."""
    runs = [("baseline", "Daksh"), ("baseline_seed1", "Daksh")]
    for owner, (param, values) in TUNING_PLAN.items():
        runs += [(f"{param}_{v}", owner) for v in values]
    runs += [("aug_none", "Jasper"), ("aug_limited", "Jasper"), ("final", "Philip")]
    return runs


def run_status(split="val", rehearsal=False):
    """Which planned runs are trained, evaluated and have figures in the shared folder.

    Slides and the report need the same figures for every run, so this shows at a glance
    whose results are still missing.
    """
    import pandas as pd
    root = output_root(rehearsal)
    rows = []
    for name, owner in planned_runs():
        train_figs = root / "figures" / name
        eval_figs = root / "figures" / f"{name}_{split}"
        rows.append(dict(run=name, owner=owner,
                         trained=(root / "metrics" / f"{name}__train.json").exists(),
                         evaluated=(root / "metrics" / f"{name}__{split}.json").exists(),
                         training_plots=len(list(train_figs.glob("*.png"))) if train_figs.is_dir() else 0,
                         eval_plots=len(list(eval_figs.glob("*.png"))) if eval_figs.is_dir() else 0))
    return pd.DataFrame(rows).set_index("run")


def seed_noise(df, baseline="baseline", repeat="baseline_seed1", metric="mAP50_95"):
    """How far the metric moves when only the random seed changes, or None if not measured.

    A tuning run whose gap to the baseline is smaller than this cannot be told apart
    from luck. One repeat is a rough estimate, so treat gaps near it as a tie.
    """
    if baseline in df.index and repeat in df.index:
        return abs(float(df.loc[repeat, metric]) - float(df.loc[baseline, metric]))
    return None


def plot_comparison(df, title, baseline="baseline", metric="mAP50_95", save_to=None, noise=None):
    """Horizontal dot plot of one metric per run, baseline marked in orange.

    With `noise` (from seed_noise), a grey band marks baseline +/- noise: runs inside it
    are not distinguishable from the baseline.
    """
    import matplotlib.pyplot as plt
    data = df[metric].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(data) + 1.4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    if noise and baseline in data.index:
        centre = float(data[baseline])
        ax.axvspan(centre - noise, centre + noise, color="#d9d8d3", alpha=0.5, zorder=0, lw=0)
    y = range(len(data))
    colors = ["#eb6834" if n == baseline else "#2a78d6" for n in data.index]
    ax.hlines(y, 0, data.values, color="#d9d8d3", linewidth=1.5, zorder=1)
    ax.scatter(data.values, y, c=colors, s=70, zorder=2, edgecolors="#fcfcfb", linewidths=2)
    for yi, value in zip(y, data.values):
        ax.annotate(f"{value:.3f}", (value, yi), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(list(y), data.index, fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, max(float(data.max()) * 1.18, 1e-3))
    label = metric
    if noise and baseline in data.index:
        label += f"    (grey band: baseline +/- seed noise of {noise:.3f})"
    ax.set_xlabel(label, fontsize=9, color="#52514e")
    ax.tick_params(axis="x", labelsize=8, colors="#52514e", length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d3")
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=160, facecolor=fig.get_facecolor())
    return fig


def fit_report(name, rehearsal=False):
    """Compare train and val loss curves: the evidence for underfit or overfit."""
    import pandas as pd
    df = pd.read_csv(output_root(rehearsal) / "runs" / name / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    val = df["val/box_loss"] + df["val/cls_loss"]
    train = df["train/box_loss"] + df["train/cls_loss"]
    best = int(val.idxmin())
    out = dict(epochs_run=len(df), best_epoch=int(df["epoch"][best]),
               final_train_loss=round(float(train.iloc[-1]), 4),
               final_val_loss=round(float(val.iloc[-1]), 4),
               lowest_val_loss=round(float(val.min()), 4),
               val_loss_rise_after_best=round(float(val.iloc[-1] - val.min()), 4),
               train_val_gap=round(float(val.iloc[-1] - train.iloc[-1]), 4))
    rising = out["val_loss_rise_after_best"] > 0.05 * out["lowest_val_loss"]
    still_falling = best >= len(df) - 2 and len(df) > 1
    out["reading"] = ("val loss rose after its lowest point: signs of overfitting" if rising else
                      "val loss still falling at the end: likely underfit, train longer" if still_falling else
                      "val loss flattened with train loss: reasonable fit")
    return out


def yolo_boxes(label_path, width, height):
    rows = []
    if Path(label_path).exists():
        for line in Path(label_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                c, x, y, w, h = map(float, line.split()[:5])
                rows.append((int(c), (x - w / 2) * width, (y - h / 2) * height,
                             (x + w / 2) * width, (y + h / 2) * height))
    return rows


def score_images(weights, root=DATASET, split="val", conf=0.25, iou=0.5, imgsz=None, limit=None):
    """Mark each image correct or wrong by matching predictions to the true boxes."""
    from ultralytics import YOLO
    root = Path(root)
    model = YOLO(str(weights))
    images = images_in(root / "images" / split)[:limit]
    scored = []
    for img in images:
        kw = {"imgsz": imgsz} if imgsz else {}
        result = model.predict(str(img), conf=conf, verbose=False, **kw)[0]
        height, width = result.orig_shape
        truth = yolo_boxes(root / "labels" / split / f"{img.stem}.txt", width, height)
        preds = [(int(c), *b, float(s)) for c, b, s in zip(result.boxes.cls.tolist(),
                                                           result.boxes.xyxy.tolist(),
                                                           result.boxes.conf.tolist())]
        unmatched, problems = list(truth), []
        for p in sorted(preds, key=lambda r: -r[5]):
            best, best_iou = None, iou
            for t in unmatched:
                overlap = _iou(t[1:5], p[1:5])
                if overlap >= best_iou:
                    best, best_iou = t, overlap
            if best is None:
                problems.append(f"false alarm: {CLASS_NAMES[p[0]]}")
            else:
                unmatched.remove(best)
                if best[0] != p[0]:
                    problems.append(f"{CLASS_NAMES[best[0]]} called {CLASS_NAMES[p[0]]}")
        problems += [f"missed {CLASS_NAMES[t[0]]}" for t in unmatched]
        scored.append(dict(image=img, truth=truth, preds=preds, problems=problems))
    return scored


def show_examples(scored, correct=True, n=6, save_to=None):
    """Grid of correct or failed images. Green = true box, orange = prediction."""
    import cv2
    import matplotlib.pyplot as plt
    picks = [s for s in scored if bool(s["problems"]) != correct]
    picks = sorted(picks, key=lambda s: -len(s["problems"]))[:n]
    if not picks:
        print("No images in this group.")
        return None
    cols = min(3, len(picks))
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, s in zip(axes.ravel(), picks):
        image = cv2.cvtColor(cv2.imread(str(s["image"])), cv2.COLOR_BGR2RGB)
        thick = max(2, image.shape[1] // 300)
        for c, x1, y1, x2, y2 in s["truth"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 160, 60), thick)
        for c, x1, y1, x2, y2, score in s["preds"]:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (235, 104, 52), thick)
            cv2.putText(image, f"{CLASS_NAMES[c]} {score:.2f}", (int(x1), max(14, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, image.shape[1] / 1100, (235, 104, 52), thick)
        ax.imshow(image)
        ax.set_title("; ".join(s["problems"][:2]) if s["problems"] else "all boxes correct", fontsize=9)
    fig.suptitle("Correct detections" if correct else "Failed detections", fontsize=12)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=130)
    return fig
