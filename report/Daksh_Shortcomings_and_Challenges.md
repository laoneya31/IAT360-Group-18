# Shortcomings and challenges (Daksh)

Everything below happened during the project. Numbers come from the baseline and batch-size runs; see
[the training report](Daksh_Baseline_and_BatchSize.md) for the full results.

## Challenges and how we addressed them

**The default optimizer ignored the learning rate.** YOLO's default `optimizer=auto` chooses its own learning rate and
logs that it is ignoring `lr0`, so tuning `lr0` would have trained the same model three times. We read the selection logic
in the Ultralytics source, found it resolves to AdamW with `lr0=0.001` for a dataset our size, and wrote those values into
the shared baseline. Behaviour stayed the same, and `lr0` became a real hyperparameter.

**Batch size is cancelled out by gradient accumulation.** YOLO accumulates gradients up to a nominal batch of 64, so the
weights update about every 64 images whatever `batch` is set to. We kept this default so the comparison changes one variable
only, and recorded the effective batch for each run. The measurements confirmed it: all four runs had an effective batch of
64, and no batch size moved mAP@0.5:0.95 by more than the run-to-run noise (0.0115). Batch size changed speed (41.1 to 49.3
minutes) more than accuracy.

**Telling a real improvement from luck.** One run per setting cannot show whether a 0.01 difference is real. We retrained
the baseline with only the random seed changed and used the gap (0.0115 mAP@0.5:0.95) as the noise level. It changed the
conclusion: three batch sizes and the best learning rate (`lr0=0.0003`, +0.009) looked different from the baseline, but
all fall within the noise.

**YOLO would not import after installing from the cloned repo.** Cloning into `/content/ultralytics` hid the installed
package, because `/content` is on Python's search path. We found this by running the setup on a fresh Colab runtime, and
fixed it by cloning into a differently named folder and doing a normal install.

**The first accuracy figure was meaningless.** YOLO validates at a near-zero confidence so it can compute mAP, which filled
the confusion matrix with thousands of low-score boxes. We now measure mAP at that default and accuracy at confidence 0.25,
and checked the result against a separate per-image matcher (0.817 and 0.825 on a practice model).

**A duplicate copy inside a public dataset.** The Kaggle freshness download contains a full second copy of itself in a
nested folder. A recursive file search would have put the same photos in train and test and inflated our scores. We use
one copy only, and `validate_dataset` checks for identical images across splits.

**The dataset changed after tuning had started.** The shared dataset was re-uploaded after the first two learning-rate runs
had trained on the earlier version. Nothing failed, but those runs were scored on a different validation set (479 fruit
instead of 445), so they could not be compared with the baseline. We caught it by comparing the number of validation boxes
recorded in each run's metrics, and the runs were retrained on the current data. `fetch_dataset` now replaces the local copy
whenever the zip on Drive changes, and refuses an empty zip.

**Colab lost sight of files on the shared Drive.** After the baseline finished, Colab reported `baseline.pt` and its metric
files as missing, even after remounting Drive, although they were on Drive (and had been loaded a minute earlier). This
happens with folders shared from another account. Re-running training would have cost 45 minutes and overwritten the
result, because a run is skipped only if its weights file is found. We loaded the same files by their Drive file id
instead, and checked the published results against Drive before committing them.

**Default colour augmentation conflicts with our task.** The default `hsv_s=0.7` shifts saturation heavily, but colour is
one of the main signs of rot. Jasper's augmentation experiment tests limited colour jitter against the defaults.

**Keeping four members' results comparable.** With runs spread across four accounts, small setup differences could
explain a "better" result. We pinned the YOLO version and put the baseline, metrics and output layout in one shared script.

**Format errors surface late.** Many label mistakes do not crash YOLO, they only lower accuracy. We wrote a validator and
proved it by planting six kinds of mistake in a copy of the practice set. On the real dataset it found no errors and two
warnings (one duplicate box, and one near-identical image pair across train and val).

**Colab limits.** Free sessions disconnect and files on the machine are lost, so results save to the shared Drive folder,
finished runs are skipped on re-run, and interrupted runs can resume from the last checkpoint. Each run took 40 to 49
minutes on a T4, so the group's 17 runs needed about 13 GPU hours across four accounts.

## Shortcomings

**Measured on the validation set (baseline):**

- **Rotten bananas.** The weakest class: mAP@0.5:0.95 0.658 and recall 0.862, with 10% of rotten bananas missed. Boxes in
  dark, overlapping bunches are loose, and our labelling rule (box a banana only if it is visible end to end) leaves
  ambiguous cases.
- **False alarms on bananas.** 48 false alarms in total, 81% of them bananas (48% fresh, 33% rotten).
- **Rotten fruit called fresh.** 3 rotten fruit were called fresh and 9 were missed, out of 240 rotten fruit. 95% of rotten
  fruit were flagged as rotten of some type. The cross-fruit error was shrivelled yellow rotten apples called fresh oranges.
- **Slight overfitting at the end of training.** After epoch 46, training loss kept falling while validation loss stopped
  improving, and the train-val gap widened from 0.28 to 0.35. Early stopping and the best checkpoint limit the damage, but
  training longer at the same settings would not help.

**Seen outside the validation set:**

- **Domain gap.** Composite scenes have clean cut-out edges and simple backgrounds, unlike real bins. On a real photo the
  model never trained on, it called an apple and a lemon "fresh orange". The validation scores (mAP@0.5:0.95 0.834) are
  likely optimistic for real store photos, which the final evaluation on our own photos will measure.
- **Three fruits only.** Anything else is out of scope, and the lemon above shows it gets labelled as one of our classes.

**Expected from the data design, not measured:**

- **Early rot.** Labels are binary, but rot is gradual. Early bruising is subtle and depends on the annotator's judgement.
- **Hidden rot.** A camera only sees one side of a fruit, so rot facing away or inside a bag cannot be detected.
- **Lighting.** Store lighting changes colour, which the model relies on.
- **One run per setting.** Tuning conclusions rest on single runs and one noise estimate from two seeds. Several seeds per
  setting would give a firmer answer, at several times the GPU cost.
- **Fixed 50 epochs.** Batch 8, 32 and 64 were still improving at epoch 50, so the batch comparison favours settings that
  converge quickly.
