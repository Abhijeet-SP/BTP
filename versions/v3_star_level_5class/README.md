# v3 — 5 classes, one per star

**Previous version:** [v2](../v2_roberta_large) — a 3x bigger model gained only +0.6 points, so size is not the limit.

## What changed
The task itself. Instead of 3 buckets, the model predicts the **star rating**:

| Class | Stars |
|---|---|
| strongly_negative | 1★ |
| negative | 2★ |
| neutral | 3★ |
| positive | 4★ |
| strongly_positive | 5★ |

This predicts **intensity**, not just direction, which is what separates this project from standard
positive/negative sentiment work. Model and settings are identical to v1 (roberta-base, lr 2e-5, batch 32,
2 epochs); only the labels and the data folder (`data5/`) differ: 40,000 reviews per star → 160k train / 20k val / 20k test.

**New metrics.** With ordered labels, predicting 4★ for a 5★ review is a small error while predicting 1★ is a large one,
so accuracy alone is misleading. Every model now also reports:
- **off-by-one accuracy** — share of predictions within one star
- **MAE in stars** — average size of the error
- **quadratic weighted kappa (QWK)** — the standard agreement measure for ordinal ratings

## Results (balanced test, 20k)
| Model | Accuracy | Macro-F1 | Off-by-one | MAE (stars) | QWK |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.580 | 0.576 | 0.916 | 0.527 | 0.806 |
| **RoBERTa-base (5-class)** | **0.646** | **0.643** | **0.959** | **0.402** | **0.876** |

Natural mix + prior correction: accuracy **0.828**, MAE **0.216 stars**, QWK **0.918**.

Per class: 1★ 0.711 · 2★ 0.529 · 3★ 0.550 · 4★ 0.623 · 5★ 0.802.

## What we learned
- Exact accuracy falls (0.827 → 0.646) **because the task is finer, not because the model is worse**:
  96% of predictions are within one star and the average error is 0.4 stars.
- Fine-tuning still beats TF-IDF clearly on the same task: +6.6 accuracy points, +7 QWK points.
- The extremes are easy, the middle (2★, 3★) is hard — exactly where star labels and text disagree.
- **One model can serve both tasks.** Summing the star probabilities into 3 classes and correcting for the
  5-class training prior (40/20/40) gives macro-F1 0.820 on the original 3-class test set, within 0.8 points
  of the dedicated v1 model. Without the correction it drops to 0.781, almost all of it neutral recall.

**This motivated v4:** the weak middle classes are where mislabelled reviews concentrate.

## Files
`notebook.ipynb` · `results/`.
Reproduce: `python prepare_data.py --classes 5 --per-class 40000`, then the notebook.
