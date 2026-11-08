#!/usr/bin/env python3
"""De-duplicate the classification source and lay it out as the project's six classes.

    python scripts/build_dataset.py              # hash, dedupe, split, write files
    python scripts/build_dataset.py --dry-run    # statistics only, writes nothing

Source (git-ignored): data/raw/fruits-fresh/, an ImageFolder tree whose class
directories are named freshapples / freshbanana / rottenoranges / ... The shipped
archive contains its tree twice (train/... and dataset/train/...), so byte-identical
copies are dropped by content hash before splitting.

Outputs (data/processed/, git-ignored)
  {train,val,test}/<class>/*.<ext>  ImageFolder-ready images
  manifest.csv                      one row per kept image
  stats.json                        machine-readable counts
  REPORT.md                         human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "fruits-fresh"
OUT = ROOT / "data" / "processed"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The team's plan merges "stale" into "rotten".
STATE_ALIASES = {"fresh": "fresh", "stale": "rotten", "rotten": "rotten"}
FRUITS = {"apple", "banana", "orange"}
TARGET_CLASSES = [
    "fresh_apple", "fresh_banana", "fresh_orange",
    "rotten_apple", "rotten_banana", "rotten_orange",
]
SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def parse_dirname(name: str) -> tuple[str, str] | None:
    """'freshapples' / 'fresh_apple' / 'stale bananas' -> ('fresh', 'apple'), ..."""
    s = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    for prefix in ("fresh", "stale", "rotten"):
        if s.startswith(prefix) and len(s) > len(prefix):
            return STATE_ALIASES[prefix], s[len(prefix):].rstrip("s")
    return None


def label_for(path: Path) -> tuple[str, str] | None:
    """Find the class folder among the file's ancestors (closest first)."""
    for part in reversed(path.relative_to(SOURCE).parts[:-1][-4:]):
        parsed = parse_dirname(part)
        if parsed:
            return parsed
    return None


def digest_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_counts(n: int) -> dict[str, int]:
    n_val = max(1, round(n * FRACTIONS["val"]))
    n_test = max(1, round(n * FRACTIONS["test"]))
    if n - n_val - n_test < 1:  # tiny-class guard
        n_val = n_test = 0
    return {"train": n - n_val - n_test, "val": n_val, "test": n_test}


def main() -> None:
    global SOURCE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data",
                    help="folder with raw/fruits-fresh (default: ./data)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()
    SOURCE = args.data_root / "raw" / "fruits-fresh"
    OUT = args.data_root / "processed"

    if not SOURCE.exists():
        sys.exit(f"missing source tree: {SOURCE}\n"
                 "put the sriramr ImageFolder export there (train/, test/, dataset/).")

    # ---------- pass 1: label every image ----------
    candidates: list[tuple[Path, str]] = []
    excluded = Counter()
    all_files = [p for p in SOURCE.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    print(f"[source] {len(all_files)} image files under {SOURCE}", flush=True)
    for n, path in enumerate(all_files, 1):
        parsed = label_for(path)
        if not parsed:
            excluded["unlabelled"] += 1
            continue
        state, fruit = parsed
        if fruit not in FRUITS:
            excluded[f"out_of_scope:{fruit}"] += 1
            continue
        candidates.append((path, f"{state}_{fruit}"))
        if n % 5000 == 0:
            print(f"  scanned {n}/{len(all_files)}", flush=True)
    print(f"[source] {len(candidates)} in-scope images", flush=True)

    # ---------- pass 2: de-duplicate by content (the tree ships twice) ----------
    seen: set[str] = set()
    by_identity: dict[tuple[str, int], str] = {}  # (name, size) -> digest; the tree ships twice
    deduped: list[tuple[Path, str, str]] = []  # path, class, digest
    dupes = 0
    for n, (path, cls) in enumerate(sorted(candidates, key=lambda c: c[0].as_posix()), 1):
        ident = (path.name, path.stat().st_size)
        digest = by_identity.get(ident)
        if digest is None:
            digest = digest_file(path)
            by_identity[ident] = digest
        if digest in seen:
            dupes += 1
            continue
        seen.add(digest)
        deduped.append((path, cls, digest))
        if n % 5000 == 0:
            print(f"  hashed {n}/{len(candidates)}", flush=True)
    print(f"unique images: {len(deduped)}  ·  duplicate copies dropped: {dupes}", flush=True)

    # ---------- stratified split ----------
    rng = random.Random(SEED)
    by_class: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for row in deduped:
        by_class[row[1]].append(row)

    assigned, plan = {}, {}
    for cls in TARGET_CLASSES:
        items = sorted(by_class.get(cls, []), key=lambda r: r[2])
        rng.shuffle(items)
        counts = split_counts(len(items))
        assigned[cls] = counts
        cursor = 0
        for split in SPLITS:
            for row in items[cursor:cursor + counts[split]]:
                plan[row[0]] = (split, cls, row[2])
            cursor += counts[split]

    # ---------- write the tree ----------
    rows: list[list[str]] = []
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            for cls in TARGET_CLASSES:
                (OUT / split / cls).mkdir(parents=True, exist_ok=True)
        ordered = sorted(plan.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0].as_posix()))
        for n, (path, (split, cls, digest)) in enumerate(ordered, 1):
            target = OUT / split / cls / f"{digest[:16]}{path.suffix.lower()}"
            if not target.exists():
                shutil.copyfile(path, target)
            rows.append([target.relative_to(OUT).as_posix(), cls, cls.split("_")[1],
                         cls.split("_")[0], split, "fruits-fresh", digest,
                         path.relative_to(SOURCE).as_posix()])
            if n % 3000 == 0:
                print(f"  wrote {n}/{len(plan)}", flush=True)
        rows.sort(key=lambda r: (r[4], r[1], r[0]))
        with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["filepath", "class", "fruit", "state", "split", "source", "md5", "original_member"])
            wr.writerows(rows)

    # ---------- report ----------
    split_totals = Counter(r[4] for r in rows) if rows else Counter()
    stats = {
        "seed": SEED,
        "split_fractions": FRACTIONS,
        "classes": TARGET_CLASSES,
        "per_class": {c: assigned.get(c, {}) for c in TARGET_CLASSES},
        "per_class_total": {c: sum(assigned.get(c, {}).values()) for c in TARGET_CLASSES},
        "split_totals": dict(split_totals),
        "total_images": len(rows) if rows else len(deduped),
        "duplicates_removed": dupes,
        "excluded_out_of_scope": dict(excluded),
    }
    if not args.dry_run:
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    lines = [
        "# Freshness dataset — build report",
        "",
        f"Images kept: **{stats['total_images']}** · duplicate copies dropped: **{dupes}** · "
        f"out-of-scope images skipped: **{sum(excluded.values())}**",
        "",
        f"Source: `{SOURCE.relative_to(ROOT).as_posix()}/`",
        "",
        "| class | train | val | test | total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cls in TARGET_CLASSES:
        a = assigned.get(cls, {})
        lines.append(f"| {cls} | {a.get('train', 0)} | {a.get('val', 0)} | {a.get('test', 0)} | {sum(a.values())} |")
    lines += ["", "## Skipped (not an apple, banana or orange)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(excluded.items())] or ["- none"]
    report = "\n".join(lines) + "\n"
    if not args.dry_run:
        (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
