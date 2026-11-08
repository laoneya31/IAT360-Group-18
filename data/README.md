# Data pipeline (Philip)

Builds the course-layout tree that Colab / `fruit_yolo` load:

```
datasets/fruit_freshness/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Public raw images stay out of git. Put them under `data/raw/` (or pass `--data-root`).

## Classes
`0 fresh_apple · 1 fresh_banana · 2 fresh_orange · 3 rotten_apple · 4 rotten_banana · 5 rotten_orange`

## Sources → `data/raw/`
- `fruits-fresh/` — Kaggle freshness classification (composites)
- `rotten-fruits/` — Roboflow YOLO export (real boxes)
- `fruit-detection/` — Kaggle VOC boxes; pass `--voc-state fresh|rotten`

## Build (from repo root)

```bash
pip install numpy pillow
python scripts/build_dataset.py
python scripts/build_yolo_detection.py --voc-state fresh \
  --train-scenes 400 --val-scenes 50 --test-scenes 50
python scripts/check_yolo_dataset.py
```

If raw data already lives next to this clone (parent `../data`):

```bash
python scripts/build_dataset.py --data-root ../data   # if that script supports it
python scripts/build_yolo_detection.py --data-root ../data --voc-state fresh \
  --train-scenes 400 --val-scenes 50 --test-scenes 50
```