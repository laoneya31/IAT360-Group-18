# Fruit Freshness Detection with YOLO

IAT 360 Computer Vision Project, Group 18: Daksh, Jasper, Lavanya, Philip.

Grocery stores lose money when rotting produce goes unnoticed in display bins and spreads spoilage. We fine-tune YOLOv8 to
detect every apple, banana and orange in a cluttered scene and flag the rotten ones. This is supervised object detection
with six classes: `fresh_apple`, `fresh_banana`, `fresh_orange`, `rotten_apple`, `rotten_banana`, `rotten_orange`.

## Notebooks

All work runs in Colab. Each member owns one notebook, so nobody edits the same file.

| Owner | Notebook | Covers | Open |
|---|---|---|---|
| Daksh | [Setup, baseline, batch size](notebooks/Daksh_Setup_Baseline_BatchSize.ipynb) | Environment, test training, baseline, `batch` tuning, figure export | [Colab](https://colab.research.google.com/github/laoneya31/IAT360-Group-18/blob/main/notebooks/Daksh_Setup_Baseline_BatchSize.ipynb) |
| Philip | [Dataset, learning rate, final model](notebooks/Philip_Dataset_LearningRate_FinalModel.ipynb) | Dataset build, `lr0` tuning, final model | [Colab](https://colab.research.google.com/github/laoneya31/IAT360-Group-18/blob/main/notebooks/Philip_Dataset_LearningRate_FinalModel.ipynb) |
| Jasper | [Annotation, image size, augmentation](notebooks/Jasper_Annotation_ImageSize_Augmentation.ipynb) | Annotation review, class chart, `imgsz` tuning, bonus augmentation | [Colab](https://colab.research.google.com/github/laoneya31/IAT360-Group-18/blob/main/notebooks/Jasper_Annotation_ImageSize_Augmentation.ipynb) |
| Lavanya | [Licences, freeze, final evaluation](notebooks/Lavanya_Licences_Freeze_FinalEvaluation.ipynb) | Licence check, `freeze` tuning, final test evaluation | [Colab](https://colab.research.google.com/github/laoneya31/IAT360-Group-18/blob/main/notebooks/Lavanya_Licences_Freeze_FinalEvaluation.ipynb) |

## Order of work

1. Everyone takes 30 photos and annotates them in Roboflow following [the labelling rules](docs/LABELLING_RULES.md). Blur any faces first.
2. Jasper reviews all annotations.
3. Philip builds the dataset and uploads `fruit_freshness.zip` to `IAT360-Group 18/dataset/` on Drive.
4. Daksh trains the baseline. Results land in `IAT360-Group 18/results/`.
5. Everyone runs their tuning. All of it compares against that one baseline.
6. Philip trains the final model from the best values.
7. Lavanya runs the final evaluation on the 120-photo test set.

## Running a notebook

1. **One-time:** in Google Drive, right-click `IAT360-Group 18` in *Shared with me* and choose **Organize > Add shortcut** into My Drive.
2. Open your notebook from the table above.
3. Choose **Runtime > Change runtime type > T4 GPU**.
4. Run the cells from the top. They clone this repo and YOLO, and load the dataset from Drive.
5. Save with **File > Save a copy in GitHub**, so your outputs are visible to markers. Commit only your own notebook.

A finished run is skipped when you re-run, so a repeat never wastes GPU time. If Colab disconnects mid-run, call
`fy.train_run(name, owner, overrides=..., resume=True)`.

## Rules that keep results comparable

- Train only through `fy.train_run`, and change only your own hyperparameter.
- Do not edit `BASELINE` in `src/fruit_yolo.py` without telling the group: it would invalidate everyone's comparison.
- Run `fy.validate_dataset()` before training on any new data.
- Do not evaluate on the test set while tuning. Tuning uses `val`. `test` is for Lavanya's final evaluation only.

| Member | Hyperparameter | Baseline | Values |
|---|---|---|---|
| Philip | `lr0` | 0.001 | 0.0003, 0.003, 0.01 |
| Jasper | `imgsz` | 640 | 416, 512, 800 |
| Daksh | `batch` | 16 | 8, 32, 64 |
| Lavanya | `freeze` | none | 5, 10, 15 |

## Repository layout

```
config.yaml                     YOLO dataset config (paths are relative to this file)
src/fruit_yolo.py               Shared training, evaluation and validation code
notebooks/                      One Colab notebook per member
datasets/fruit_freshness/       YOLO folders: images/ and labels/, each with train, val, test
docs/LABELLING_RULES.md         What counts as fresh or rotten, and how to draw boxes
docs/DATA_SOURCES.md            Dataset links and licences
results/                        Final metrics, figures and screenshots for the report
weights/                        Saved models
report/                         Report outline and section drafts
```

**Dataset:** our own photos go in `images/test` and `labels/test` on GitHub, unless the licence and privacy review says
otherwise. Public datasets are not redistributed here: extract `fruit_freshness.zip` into `datasets/fruit_freshness/`. See
[data sources](docs/DATA_SOURCES.md).
