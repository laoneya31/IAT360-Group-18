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
| imgsz 416 | image size | 0.823 | -0.011 | tie | | 0.937 | 0.968 | 0.953 | 0.972 | 47 |
| imgsz 512 | image size | 0.825 | -0.009 | tie | | 0.938 | 0.950 | 0.944 | 0.935 | 52 |
| imgsz 800 | image size | 0.821 | -0.013 | worse (just) | | 0.949 | 0.926 | 0.937 | 0.933 | 51 |

## What the tuning shows

**The baseline was already close to the best this setup can do.** Of twelve tuning runs, none beat it by more than the
seed noise. Seven were ties and five were worse. The settings that hurt all did the same thing: they made it harder for
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

**Final model: the baseline settings with `lr0=0.0003`** (YOLOv8s, `batch=16`, nothing frozen, `imgsz=640`, 50 epochs,
AdamW, seed 0). Philip trained it as the run `final`.

**Why only the learning rate changed.** The group agreed before tuning that a gap inside 0.0115 is a tie, not a win. By
that rule no batch size, freeze or image size value beat the baseline, so those keep their baseline values. Combining
several values that each tied would not be justified by the evidence, and could combine their noise rather than their
gains.

**Why `lr0=0.0003`.** It was also inside the noise on mAP (+0.009), but it was the only run that improved on every main
metric at once: the highest mAP (0.843), the highest precision (0.965) and F1 (0.965), the best rotten-banana score
(0.689) and 40% fewer false alarms (29 against 48). That fits how a smaller learning rate behaves on a pretrained model:
it changes the COCO features more gently and overfits less.

**The test set supports the choice.** On our 120 photos the final model beats a model with the baseline settings by
0.057 mAP@0.5:0.95 (0.286 against 0.229), five times the validation seed noise, and finds twice as many rotten fruit
(rotten recall 0.406 against 0.200). Caveat: the noise level was measured on validation, and each model was trained once.

**Why validation scores are so high.** YOLOv8s starts from weights that already detect fruit-like objects (COCO includes
apples, bananas and oranges), and the validation images come from the same public sources as the training images.
Validation measures how well the model learned those sources, not how it does in a real store. The test set measures
that.

## Final evaluation on our 120 photos

120 photos we collected and labelled ourselves, 30 per member, 483 fruit. Used once, after the final model was chosen.

| Metric | Final, validation | Final, test | Baseline settings, test |
|---|---|---|---|
| mAP@0.5:0.95 | 0.843 | **0.286** | 0.229 |
| mAP@0.5 | 0.980 | **0.446** | 0.359 |
| Precision | 0.965 | **0.585** | 0.560 |
| Recall | 0.964 | **0.503** | 0.348 |
| F1 | 0.965 | **0.541** | 0.430 |
| Rotten recall | 0.943 | **0.406** | 0.200 |
| Accuracy | 0.907 | **0.411** | 0.384 |

| Test class | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| fresh apple | 0.658 | 0.496 | 0.565 | 0.513 | 0.325 |
| fresh banana | 0.411 | 0.611 | 0.492 | 0.432 | 0.245 |
| fresh orange | 0.669 | 0.691 | 0.680 | 0.657 | 0.356 |
| rotten apple | 0.646 | 0.498 | 0.563 | 0.491 | 0.357 |
| rotten banana | 0.346 | 0.464 | 0.396 | 0.319 | 0.212 |
| rotten orange | 0.780 | 0.257 | 0.386 | 0.266 | 0.223 |

| Photos | Fruit | Right | Wrong class | Missed | False alarms | Images with no mistakes |
|---|---|---|---|---|---|---|
| Jasper | 72 | 58 | 6 | 8 | 56 | 13 / 30 |
| Philip | 168 | 86 | 10 | 72 | 27 | 2 / 30 |
| Daksh | 150 | 60 | 8 | 82 | 22 | 6 / 30 |
| Lavanya | 93 | 38 | 15 | 40 | 48 | 3 / 30 |
| **Total** | **483** | **242** | **39** | **202** | **153** | **24 / 120** |

**The model is much worse on our photos than on validation (0.843 to 0.286).** Training and validation images are
mostly one or a few fruit on plain backgrounds; our photos are store displays, crowded bowls, bagged fruit and mixed
lighting.

- **Most errors are missed fruit.** 202 of 483 fruit (42%) were missed, only 39 got the wrong class. Misses pile up in
  crowded scenes: in one store display (`daksh_012`) all 21 fruit were missed. Jasper's photos, with fewer and larger
  fruit, did best.
- **Bananas cause most false alarms** (93 of 153), mostly bunches: some labels box a whole bunch and others each banana,
  so one fruit counts as both a miss and a false alarm. Part of this is our labelling.
- **Rotten called fresh happened 28 times** (rotten banana 13, apple 8, orange 7). Training rot is mostly heavy rot, so
  early rot reads as fresh. Rotten orange recall is lowest (0.257).
- **Some error is ours:** two rotten-apple photos labelled as oranges and five banana bunches boxed as one fruit in
  Lavanya's set.

**What this says.** The model works on clean close-ups, including heavily mouldy fruit, but is not ready for a store: it
misses too many fruit in real displays and too much early rot. The fix is in the data, not the settings: real store
photos with many fruit per image, early-rot examples, and one agreed rule for banana bunches.
