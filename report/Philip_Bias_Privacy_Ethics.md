# Bias, privacy and ethics (Philip)

Numbers below come from the shared validation set (268 images, 445 fruit) and the group’s tuning runs. Seed noise on mAP@0.5:0.95 is **0.0115** (Daksh’s `baseline` vs `baseline_seed1`). Full tables: Daksh (baseline / batch), Jasper (imgsz / augmentation), Lavanya (freeze / model analysis), Philip (`lr0`).

## Bias in the dataset

**Fruit types and cultivars.** We only label apples, bananas and oranges. Any other produce gets ignored, or it gets misclassified if it looks close enough to one of ours. Public sources also lean toward the usual looks: bright red apples, yellow Cavendish bananas, navel oranges. So an unusual cultivar might get called rotten when it's fresh, or get missed when it's actually rotten. Daksh’s baseline already shows one cross-fruit failure of this kind: shrivelled yellow rotten apples called fresh orange. On an unseen store photo the same model called a lemon “fresh orange.”

**Fresh vs rotten balance / rotten bananas.** Late-stage rot (mold, heavy browning) is much easier to find photos of than an early bruise, so the model sees more of the obvious cases. Across **every** tuning run, rotten banana stays the weakest class (mAP@0.5:0.95 roughly **0.60–0.69**; other classes mostly **0.69–0.93**). Baseline recall for rotten banana is **0.862** and mAP@0.5:0.95 is **0.658**; mean rotten recall is **0.926**, and **95%** of rotten fruit are flagged as some rotten class. Jasper’s augmentation runs make the same point: rotten banana scored **0.54** with no colour augmentation and **0.67** with softer colour. Lavanya’s `freeze=10` drops rotten banana further (to **0.603**). Philip’s `lr0=0.0003` is the best rotten-banana score among the main runs (**0.689**) but still far behind the other classes. No hyperparameter fixed this, which points at the **data** (few early-rot bananas, inconsistent bunch boxes in public Roboflow data, hard dark overlapping shapes) rather than at training settings alone.

**Banana bunches and label rules.** Public Roboflow data sometimes boxes every banana and sometimes the whole bunch. Our rule is one tight box per fruit, and bananas must show end-to-end (except the stem). Daksh notes that many false alarms are bananas in dark overlapping bunches that annotators skipped under that visibility rule — so part of the “error” is label ambiguity. Peeled or cut fruit are never labelled, so the model learns to ignore them by design.

**Synthetic and composite scenes.** A lot of train/val is cut-outs and composites, not real bins. That favours clean edges and under-represents bags, price tags, occlusion and store lighting. Our **120 own photos** are held-out **test** so that gap is measured. Lavanya’s test eval shows the drop: final model mAP@0.5:0.95 **0.286** on test vs much higher on val (and still better than baseline settings on the same test set, **0.229**).

**Geography and store context.** Local shops plus web photos don't cover every region, season or packaging style.

**What we did, and what we'd do next.** Unified labelling rules and Jasper’s review; colour jitter kept soft so rot colour is not washed out (Jasper’s softer-colour run found slightly more rotten fruit, **95%** vs baseline **93%**, without beating mAP). Jasper’s `imgsz=416` found the most rotten fruit on val (**97%**) but did not beat 640 on mAP, so we kept **640**. Next steps: more early-rot and banana examples, consistent bunch labelling in any new public data we add, more real bin photos in train/val, and per-scene metrics (bag / pile / lighting).

## Privacy

- Blur faces and name badges before anything goes to Roboflow, GitHub or Drive.
- Avoid receipts, loyalty cards and security screens. Crop to the produce where possible.
- For public datasets, we link to them when we can't redistribute. The train/val zip lives on Drive, and our test images go on GitHub after a privacy review (`docs/DATA_SOURCES.md`).
- No API keys in the repo. We use Colab Secrets.
- A real store camera would need short retention, access control and no identity features. This project stays offline and only uses curated images.

## Ethical considerations

**Errors (cost of mistakes).**
- *Missed rot / called fresh:* spoilage spreads; customers get bad fruit. Baseline: **3** called fresh, **9** missed. `lr0=0.0003`: **1** called fresh, **10** missed. Lavanya’s `freeze=5` (tie on mAP) had the best rotten-oriented counts among freeze runs: **0** called fresh, **5** missed, rotten recall **0.959** — still one run, and seed alone moves rotten recall by ~0.018.
- *False rotten / false alarms:* good fruit discarded → waste and distrust. Baseline **48** false alarms; Daksh notes bananas dominate them. `lr0=0.0003` cuts false alarms to **29**. Raising `lr0` to **0.01** nearly doubles them (**91**). Freezing hard (`freeze=15`) raises them to **65**.
- Keep a human in the loop for any operational use. Prefer reporting **rotten recall and false alarms**, not only mAP.

**Surveillance.** Overhead “spoilage” cameras can be misused for customer or staff tracking. Scope to produce regions only.

**Food waste.** The goal is less waste and safer shelves, but only if rotten recall stays high and false alarms stay low enough that staff trust alerts. Soft colour augmentation and smaller image size slightly raised rotten-fruit found without clearly winning on mAP; the final choice (`lr0=0.0003`, keep batch 16 / imgsz 640 / no freeze) is the run that improved precision and cut false alarms without hurting rotten recall.

**Labour.** Assist produce staff; do not replace judgement. Early rot is subjective — shared rules + Jasper’s review reduce four different label policies. Freezing the whole backbone (Lavanya) underfits rot textures that COCO never learned, which is another reason not to ship a “faster but frozen” model for this task.
