# Introduction and presentation scripts (Lavanya)

Report section: Introduction. Presentation sections: 1 (problem and use case, 0:45) and 3b (worked and failed examples,
0:40). The freeze tuning results are in [Lavanya_Freeze.md](Lavanya_Freeze.md).

## Introduction

Grocery stores lose money when rotting produce goes unnoticed in display bins. One spoiled fruit spreads mold and
ethylene to the fruit around it, and shoppers who see it trust the whole display less. Today staff catch this by checking
bins by hand, which is slow, happens only a few times a day, and depends on who is looking. We train a model that looks
at a cluttered scene of apples, bananas and oranges, draws a box around every fruit, and labels each one as fresh or
rotten. This is a **supervised object detection** problem: the model learns from images where every fruit is boxed and
labelled with one of six classes (`fresh_apple`, `fresh_banana`, `fresh_orange`, `rotten_apple`, `rotten_banana`,
`rotten_orange`). Detection, rather than classification of whole images, is needed because a real bin holds many fruit
at once, and staff need to know which one to remove. The **target audience** is produce staff and store managers, who
would get an alert to remove or mark down a flagged fruit, and back-room teams who sort deliveries before they reach the
shelf. The **target devices** are a fixed camera above a produce display or a sorting table, with the model running on a
small edge computer in the store (for example an NVIDIA Jetson) or on a store server. We fine-tune YOLOv8s because it is
small and fast enough to run in real time on such hardware. Since a missed rotten fruit costs more than a fresh one sent
for a second look, we pay most attention to **rotten-class recall**.

## Dataset: Lavanya's 30 test images

The 30 images are openly licensed photos of apples, bananas and oranges
(fresh, rotten and mixed scenes) from Flickr and Wikimedia Commons, found through Openverse and credited in
[`docs/TEST_IMAGE_CREDITS.md`](../docs/TEST_IMAGE_CREDITS.md). Lavanya annotated them in a private Roboflow project with
the group's six classes and exported them in YOLOv8 format into `datasets/fruit_freshness/{images,labels}/test`.
93 boxes: fresh_apple 21, fresh_banana 13, fresh_orange 29, rotten_apple 16, rotten_banana 6, rotten_orange 8.

**Known label issues.** These lower the measured test score on the affected classes:

- `lavanya_02` (4 boxes, titled "Rotten apples" at its source) and `lavanya_30` are rotten apples labelled
  `rotten_orange` and `fresh_orange`.
- In `lavanya_14`, `15`, `16`, `17` and `19` one box covers a whole banana bunch, against the rule of one box per banana.

## Presentation script: 1. Problem and use case (0:45)

> Hi, I'm Lavanya. Picture a grocery store's apple bin. One rotten apple at the bottom starts to mold, and within a day
> the apples touching it are going too. Stores lose money on this every week, and right now the only defence is staff
> checking bins by hand, which is slow and inconsistent.
>
> Our project trains a model to do that check. A camera above a produce display, or above a back-room sorting table,
> sees a messy mix of apples, bananas and oranges. The model finds every single fruit, draws a box around it, and says
> whether it's fresh or rotten. That's supervised object detection with six classes.
>
> The people who'd use it are produce staff and store managers: they get a flag on exactly which fruit to pull or mark
> down. And because missing a rotten fruit is the expensive mistake, the number we care about most is how many rotten
> fruit the model catches: rotten recall.

**Slide:** "Problem and use case", with a photo of a mixed fruit bin and the rotten fruit circled, and three lines:
*Who: produce staff and managers. Where: a camera over displays or sorting tables. What: box every fruit, flag the rotten
ones.*

## Presentation script: 3b. Worked and failed examples (0:40)

Outline, using the example images from the final evaluation on the test set:

1. A correct detection in a hard scene, and what the model got right (5 s).
2. Two or three failures and their causes: bagged fruit, early rot that looks fresh, poor lighting, or overlapping
   bananas (25 s).
3. What the failures say about the model's bias: it learned mostly from clean public photos rather than real store bins
   (10 s).
