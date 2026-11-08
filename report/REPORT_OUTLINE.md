# Report outline

Maps each rubric item to its owner. The final report is a PDF on Canvas, and every link in it must work.

| Section | Marks | Owner | Must include |
|---|---|---|---|
| Overall structure and citations | 0.5 | All | Project name, group name, student names and numbers, GitHub link, citations for all data, code and tools |
| Introduction | 0.5 | Lavanya | One paragraph: the problem, type of vision problem, supervised, use cases, target audience, target devices. Written: [introduction](Lavanya_Introduction_and_Presentation.md) |
| Dataset report | 2.5 | Jasper | How data was collected and created, labels, examples of new images with annotations, annotation tool links |
| Bias and privacy | part of dataset | Philip | Dataset bias and how it could be solved, privacy issues in detail |
| Training report: baseline | 2.5 | Daksh | Training and evaluation screenshots, metrics, underfit or overfit with concrete fixes. Written: [baseline and batch size](Daksh_Baseline_and_BatchSize.md) |
| Training report: examples | part of training | Lavanya | Correct and incorrect test images, why the model failed |
| Tuning comparison | part of training | All | Each member's three values against the baseline |
| Best model | part of training | Lavanya, Philip | Which combination won, with justification |
| Shortcomings and challenges | part of training | Daksh | Written: [shortcomings and challenges](Daksh_Shortcomings_and_Challenges.md) |
| Ethical considerations | part of training | Philip | Effects of errors, surveillance in stores, food waste |

## Submission checklist

- [ ] 30 new images and labels per person (120 in total) in the YOLO folders
- [ ] `config.yaml` correct, and a fresh clone runs
- [x] Baseline saved and evaluated: accuracy, precision, recall, F1 (`weights/baseline.pt`, `results/metrics/baseline__val.json`)
- [ ] Correct and incorrect example images shown
- [ ] Each member's 3 hyperparameter values compared with the baseline
- [ ] All steps visible in GitHub, with screenshots in `results/screenshots/`
- [ ] Notebooks saved with their outputs
- [ ] Links in the PDF checked in a private browser window
