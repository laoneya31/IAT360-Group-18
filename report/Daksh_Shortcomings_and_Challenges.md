# Shortcomings and challenges (Daksh)

Draft. The challenges below happened. Anything marked TODO needs results from the real baseline.

## Challenges and how we addressed them

**The default optimizer ignored the learning rate.** YOLO's default `optimizer=auto` chooses its own learning rate and
logs that it is ignoring `lr0`, so tuning `lr0` would have trained the same model three times. We read the selection logic
in the Ultralytics source, found it resolves to AdamW with `lr0=0.001` for a dataset our size, and wrote those values into
the shared baseline. Behaviour stayed the same, and `lr0` became a real hyperparameter.

**Batch size is partly cancelled out by gradient accumulation.** YOLO accumulates gradients up to a nominal batch of 64, so
the weights update about every 64 images whatever `batch` is set to. We kept this default so the comparison changes one
variable only, and recorded the effective batch for each run. TODO: state the measured effect from the real runs.

**YOLO would not import after installing from the cloned repo.** Cloning into `/content/ultralytics` hid the installed
package, because `/content` is on Python's search path. We found this by running the setup on a fresh Colab runtime, and
fixed it by cloning into a differently named folder and doing a normal install.

**The first accuracy figure was meaningless.** YOLO validates at a near-zero confidence so it can compute mAP, which filled
the confusion matrix with thousands of low-score boxes. We now measure mAP at that default and accuracy at confidence 0.25,
and checked the result against a separate per-image matcher (0.817 and 0.825 on a practice model).

**A duplicate copy inside a public dataset.** The Kaggle freshness download contains a full second copy of itself in a
nested folder. A recursive file search would have put the same photos in train and test and inflated our scores. We use
one copy only, and `validate_dataset` checks for identical images across splits.

**Default colour augmentation conflicts with our task.** The default `hsv_s=0.7` shifts saturation heavily, but colour is
one of the main signs of rot. Jasper's augmentation experiment tests limited colour jitter against the defaults.

**Keeping four members' results comparable.** With runs spread across four accounts, small setup differences could
explain a "better" result. We pinned the YOLO version and put the baseline, metrics and output layout in one shared script.

**Format errors surface late.** Many label mistakes do not crash YOLO, they only lower accuracy. We wrote a validator and
proved it by planting six kinds of mistake in a copy of the practice set.

**Colab limits.** Free sessions disconnect and files on the machine are lost, so results save to the shared Drive folder,
finished runs are skipped on re-run, and interrupted runs can resume from the last checkpoint.

TODO: add anything that comes up during the real training runs.

## Shortcomings

Expected from the data design, to be confirmed or removed once real results exist:

- **Domain gap.** Composite scenes have clean cut-out edges and simple backgrounds, unlike real bins. Expect lower scores on our store photos than on the public validation set.
- **Early rot.** Labels are binary, but rot is gradual. Early bruising is subtle and depends on the annotator's judgement.
- **Hidden rot.** A camera only sees one side of a fruit, so rot facing away or inside a bag cannot be detected.
- **Three fruits only.** Anything else is out of scope and may be mislabelled.
- **Lighting.** Store lighting changes colour, which the model relies on.

TODO, with real numbers:
- Baseline mAP@0.5, mAP@0.5:0.95, precision, recall, F1 and accuracy
- Rotten recall per class, and how many rotten fruit were called fresh
- Underfit or overfit, using `fy.fit_report("baseline")`, with concrete fixes
- Batch size result and whether the gap exceeds run-to-run noise
