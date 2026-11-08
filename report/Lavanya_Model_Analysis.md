# Model analysis and best model (Lavanya)

Report sections: model analysis and best-model discussion. The baseline itself (metrics, confusion matrix, fit) is
analysed in detail in [Daksh's baseline report](Daksh_Baseline_and_BatchSize.md); this section compares every run
against it. Each member's own write-up: [batch size](Daksh_Baseline_and_BatchSize.md#batch-size-tuning),
[freeze](Lavanya_Freeze.md). Source of every number: [`results/metrics/`](../results/metrics/).

## How the runs were compared

Every run fine-tunes YOLOv8s (COCO-pretrained) for 50 epochs on the same training data, with the same seed, and is
scored on the same validation set (268 images, 445 fruit). Each member changed exactly one hyperparameter and kept the
rest at the baseline, so any difference comes from that one setting.

Two runs with identical settings and only a different random seed differ by **0.0115 mAP@0.5:0.95** (`baseline` against
`baseline_seed1`). We treat that as the noise level: a run only counts as better or worse than the baseline if it moves
further than that. The main metric is mAP@0.5:0.95, because it rewards both finding a fruit and boxing it tightly. Since
a rotten fruit left in a bin is the costly mistake, we also report **rotten recall** (the share of rotten fruit found
with the right class).

## All runs against the baseline

| Run | Changed | mAP@0.5:0.95 | Gap | Reading | mAP@0.5 | Precision | Recall | F1 | Rotten recall | False alarms |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | none | **0.834** | | | 0.973 | 0.930 | 0.947 | 0.938 | 0.926 | 48 |
| baseline_seed1 | seed | 0.823 | -0.011 | noise | 0.976 | 0.928 | 0.954 | 0.941 | 0.944 | 54 |
| lr0 0.0003 | learning rate | **0.843** | +0.009 | tie | **0.980** | **0.965** | **0.964** | **0.965** | 0.943 | **29** |
| lr0 0.003 | learning rate | 0.810 | -0.024 | worse | 0.972 | 0.917 | 0.922 | 0.919 | 0.909 | 68 |
| lr0 0.01 | learning rate | 0.782 | -0.052 | worse | 0.958 | 0.910 | 0.899 | 0.905 | 0.896 | 91 |
| batch 8 | batch size | 0.825 | -0.009 | tie | 0.975 | 0.943 | 0.950 | 0.946 | 0.945 | 45 |
| batch 32 | batch size | 0.823 | -0.012 | tie | 0.973 | 0.940 | 0.970 | 0.954 | 0.958 | 36 |
| batch 64 | batch size | 0.831 | -0.003 | tie | 0.966 | 0.926 | 0.950 | 0.938 | 0.940 | 47 |
| freeze 5 | frozen layers | 0.828 | -0.007 | tie | 0.977 | 0.934 | 0.955 | 0.945 | **0.959** | 48 |
| freeze 10 | frozen layers | 0.799 | -0.035 | worse | 0.963 | 0.913 | 0.928 | 0.921 | 0.925 | 51 |
| freeze 15 | frozen layers | 0.796 | -0.038 | worse | 0.960 | 0.933 | 0.902 | 0.917 | 0.904 | 65 |

Jasper's image-size runs (416, 512 and 800) are not in this table yet.

## What the tuning shows

**The baseline was already close to the best this setup can do.** Of nine tuning runs, none beat it by more than the seed
noise. Five were ties and four were clearly worse. The settings that hurt all did the same thing: they made it harder for
the network to adjust its pretrained weights to our data.

- **Learning rate is the most sensitive setting.** Raising `lr0` from 0.001 to 0.003 or 0.01 cost 0.024 and 0.052. Large
  steps knock the COCO-pretrained weights too far before the model settles, and false alarms nearly double (48 to 91).
  Lowering it to 0.0003 gave the highest mAP of any run (0.843), but the gain (+0.009) is inside the noise.
- **Freezing more than the first few layers hurts.** Freezing the whole backbone (10) or more (15) cost about 0.035,
  three times the noise. The early layers find edges and colours that COCO already learned well, but rot is a texture
  and colour change that COCO never had to separate, so the deeper layers need to train.
- **Batch size makes no real difference.** YOLO accumulates gradients up to an effective batch of 64 in every run, so the
  weights are updated equally often whatever the batch size. All three values tied.

**Rotten bananas are the weak class in every run.** Their mAP@0.5:0.95 stays between 0.60 and 0.69, while every
other class scores between 0.69 and 0.93. Their mAP@0.5 is at least 0.91 in every run, so the model finds them; the problem is that the
boxes are loose around overlapping, dark, curved fruit, and that our labels for banana bunches are themselves
inconsistent. No hyperparameter fixed this, which points to the data rather than the training settings.

**Fit.** The baseline and the tied runs reach their best validation score between epochs 42 and 50 and level off, which
is a reasonable fit; Daksh's report shows the baseline's slight overfitting in the last four epochs, after mosaic
augmentation switches off. The clearest exception is `freeze_15`, whose validation loss was still falling at epoch 50:
with too few trainable layers it underfits.

## Best model

**Recommended final model: the baseline settings** (YOLOv8s, `lr0=0.001`, `batch=16`, nothing frozen, `imgsz=640`,
50 epochs, AdamW), unless Jasper's image-size runs show a gain larger than the noise.

**Why.** The group agreed before tuning that a gap inside 0.0115 is a tie, not a win. By that rule every hyperparameter
keeps its baseline value: no learning rate, batch size or freeze setting beat 0.834 by more than the noise. Combining
several values that each tied would not be justified by the evidence, and could combine their noise rather than their
gains.

**The one setting worth a second look is `lr0=0.0003`.** It had the highest mAP (0.843), the highest precision (0.965)
and F1 (0.965), the best rotten-banana score (0.689) and 40% fewer false alarms (29 against 48). That pattern fits how a
smaller learning rate behaves on a pretrained model: it changes the COCO features more gently and overfits less. But it is
one run, and +0.009 is smaller than the difference two seeds of the baseline produce. If there is GPU time, training
`lr0=0.0003` with a second seed would settle it; if both seeds stay above the baseline, it should replace 0.001.

**Why the baseline performs so well.** YOLOv8s starts from weights that already detect fruit-like objects (COCO includes
apples, bananas and oranges), and the validation images come from the same public sources as the training images. The
tuning mostly moved the model away from that good starting point. The real test is how the model does on our own,
messier photos, which the final evaluation measures.
