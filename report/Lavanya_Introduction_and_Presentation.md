# Introduction and presentation scripts (Lavanya)

Report section: Introduction (0.5 marks). Presentation sections: 1 (problem and use case, 0:45) and 3b (worked and failed
examples, 0:40). Model analysis and the best-model discussion follow once the freeze runs and the final model are done.

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

About 120 words, which fits 45 seconds at a normal pace.

**Slide:** title "Problem and use case"; one photo of a mixed fruit bin with a rotten fruit circled; three short lines:
*Who: produce staff, managers. Where: camera over displays or sorting tables. What: box every fruit, flag rotten ones.*

## Presentation script: 3b. Worked and failed examples (0:40)

To be filled in from the final evaluation (`examples_correct.png` and `examples_incorrect.png`). Planned structure:

1. One correct example: what the model got right in a hard scene (5 s).
2. Two or three failures, each with the reason: bagged fruit, early rot that looks fresh, poor lighting, or overlapping
   bananas (25 s).
3. What this says about the model's bias, for example that it learned from clean public photos and not real store bins
   (10 s).
