# Data sources

Owner: Lavanya. Licences were checked on each dataset page on 2026-11-08. The assignment says that if we have no sharing
rights, we link the dataset and do not share it on GitHub or as a zip.

| Dataset | Link | Used for | Licence | On GitHub? |
|---|---|---|---|---|
| Detection of Rotten Fruits (Roboflow), by "AI project", 1,182 images, 6 classes | https://universe.roboflow.com/ai-project-t3iak/detection-of-rotten-fruits | Train / val | CC BY 4.0 | Linked only |
| Fruit Images for Object Detection (Kaggle), by Muhammed Buyukkinaci | https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection | Train / val, relabelled with freshness | CC0: Public Domain | Linked only |
| Fruits Fresh and Rotten for Classification (Kaggle), by Sriram Reddy Kalluri | https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification | Composite scenes; this download has no "stale" folders, so nothing needs merging | Unknown (none stated) | **No.** Linked only |
| Our own 120 photos | `datasets/fruit_freshness/images/test` | Held-out test set | Ours | Yes, after faces, name tags and licence plates are blurred |

## Sharing decisions

- **Roboflow, CC BY 4.0.** Sharing is allowed if we credit the author and link the licence. We still link it rather than
  copy it, because the repo stays small and everyone downloads the same version.
- **Kaggle object detection, CC0.** No restrictions. We link it for the same reason. The uploader collected the images
  from the web, so CC0 is the uploader's claim, and we cite the page rather than claim the images are free of copyright.
- **Kaggle freshness, licence unknown.** With no licence stated we have no sharing rights, so we use it for training only
  and never upload its images or our composites made from them.
- **`fruit_freshness.zip`.** It contains composites from the unknown-licence dataset, so it stays on the private group
  Drive and is never put on GitHub or shared publicly. `.gitignore` already excludes it and the train/val images.
- **Our own photos.** We own them. Before upload: blur faces, name tags and licence plates, and strip GPS location from
  EXIF data. Store photos show only produce displays. If a store asks us not to photograph, we don't.

## Citations

- AI project. (2024). *Detection of Rotten Fruits Dataset* [Data set]. Roboflow Universe.
  https://universe.roboflow.com/ai-project-t3iak/detection-of-rotten-fruits. Licensed under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).
- Buyukkinaci, M. (2018). *Fruit Images for Object Detection* [Data set]. Kaggle.
  https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection. CC0: Public Domain.
- Kalluri, S. R. (2018). *Fruits fresh and rotten for classification* [Data set]. Kaggle.
  https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification.
- Jocher, G., Qiu, J., & Chaurasia, A. (2023). *Ultralytics YOLO* (Version 8) [Computer software].
  https://github.com/ultralytics/ultralytics. AGPL-3.0.
- Roboflow. (2026). *Roboflow* [Computer software]. https://roboflow.com

## Tools

| Tool | Link | Used for |
|---|---|---|
| Roboflow | https://roboflow.com | Annotation |
| Ultralytics YOLO | https://github.com/ultralytics/ultralytics | Model and training |

## Notes on these datasets

- The Kaggle freshness download contains a second copy of the data in `dataset/dataset/`. Use only one copy.
- Its images are screenshots of pre-augmented photos, so near-duplicates of one fruit exist. Rotated copies are not caught by the near-duplicate check in `validate_dataset`.
