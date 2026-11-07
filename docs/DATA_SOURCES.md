# Data sources

Owner: Lavanya. The licence column must be confirmed from each dataset page before anything is shared. If we have no
sharing rights, the assignment says to link the dataset and not share it on GitHub or as a zip.

| Dataset | Link | Used for | Licence | On GitHub? |
|---|---|---|---|---|
| Detection of Rotten Fruits (Roboflow) | https://universe.roboflow.com/ai-project-t3iak/detection-of-rotten-fruits | Train / val | TODO | TODO |
| Fruit Images for Object Detection (Kaggle) | https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection | Train / val, relabelled with freshness | TODO | TODO |
| Fruits Fresh and Rotten for Classification (Kaggle) | https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification | Composite scenes; merge "stale" into "rotten" | TODO | TODO |
| Our own 120 photos | `datasets/fruit_freshness/images/test` | Held-out test set | Ours; check store photo policy | TODO after face blurring |

## Tools

| Tool | Link | Used for |
|---|---|---|
| Roboflow | https://roboflow.com | Annotation |
| Ultralytics YOLO | https://github.com/ultralytics/ultralytics | Model and training |

## Notes on these datasets

- The Kaggle freshness download contains a second copy of the data in `dataset/dataset/`. Use only one copy.
- Its images are screenshots of pre-augmented photos, so near-duplicates of one fruit exist. Rotated copies are not caught by the near-duplicate check in `validate_dataset`.
