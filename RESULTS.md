# Results — v1 (19 Sep 2026)

Fine-tuned `roberta-base` for 3-class sentiment on Amazon Reviews 2023, **Industrial_and_Scientific**.

## Data
| | |
|---|---|
| Raw reviews | 5,183,005 (1★ 584k · 2★ 235k · 3★ 316k · 4★ 561k · 5★ 3.49M) |
| After cleaning (drop < 3 words, exact duplicates) | 4,757,146 |
| Labels | 1–2★ negative · 3★ neutral · 4–5★ positive |
| Balanced sample (reservoir sampling, seed 42) | 50,000 per class |
| Split (80/10/10, stratified) | train 120,000 · val 15,000 · test 15,000 |
| Review length | mean 44 words, median 29, p95 130 → `max_length=256` |

Raw data is heavily skewed to positive (78%), so classes were balanced; the neutral class limits the sample (305k available).

## Test-set results (15,000 reviews, balanced)
| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| TF-IDF (1–2-gram) + Logistic Regression | 0.780 | 0.780 | 0.776 | 0.690 | 0.874 |
| **RoBERTa-base fine-tuned** | **0.827** | **0.827** | **0.810** | **0.748** | **0.925** |
| Gain | +4.7 pts | +4.7 pts | +3.3 | +5.8 | +5.0 |

Fine-tuning helps most on the hardest class (neutral, +5.8 F1).

## Training (Google Colab, T4 GPU)
lr 2e-5 · batch 32 · 2 epochs (7,500 steps) · warmup 6% · weight decay 0.01 · fp16 · 41.8 min.

| Epoch | Val loss | Val accuracy | Val macro-F1 |
|---|---|---|---|
| 1 | 0.423 | 0.8285 | 0.8286 |
| 2 | 0.419 | 0.8298 | 0.8300 |

No overfitting (train loss ≈ 0.36, val 0.42). Epoch 2 adds only +0.14 pts, so the model has converged; more epochs would not help.

Figures: `results/loss_curve_roberta.png`, `results/confusion_matrix_roberta.png`, `results/confusion_matrix_tfidf.png`.

## Error analysis
Confusion matrix (row-normalized): negative→neutral 18%, neutral→negative 19%, neutral→positive 6%, positive→neutral 7%.
Negative↔positive confusion is only ~1%, so almost all errors are between **adjacent** classes.

Error rate by original star rating:

| Stars | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---|---|---|---|---|
| Error rate | 11.0% | **38.8%** | 24.9% | **34.8%** | 3.7% |

Errors concentrate on the boundary ratings (2★, 3★, 4★), where the star rating and the review text often disagree.
The model's most confident "errors" (`results/errors_sample_roberta.csv`) are mostly **label noise**. Examples:
- 1★ "These gloves work very well … I love these gloves" → predicted positive
- 5★ "One Star. It do not work?" → predicted negative
- 3★ "What a great adhesive … Use it on everything" → predicted positive

So ~0.83 is close to the ceiling that star-derived labels allow.

## Local deployment (Apple M2)
| | |
|---|---|
| Model size | 499 MB (fp32) → **249 MB (fp16)** |
| Accuracy after fp16 conversion | 0.8273 macro-F1, identical to fp32 |
| Inference speed | ~27 reviews/s (MPS GPU, batch 64) |

`python predict.py "review text"` gives a label and confidence in a few seconds, including model load.

## Reproducibility
`prepare_data.py` is deterministic (seed 42). Rerunning it locally reproduced Colab's split exactly: identical stats, and the RoBERTa test score matches to 4 decimals.

---

# v2 (in progress)

## Real-world class mix: natural-distribution test set
The balanced test set (1/3 per class) overstates how often neutral and negative reviews occur. `data/test_natural.csv` holds 20,000 reviews
in the true mix (negative 16.9% · neutral 6.4% · positive 76.7%), with no overlap with the balanced splits.

The model was trained on a uniform prior, so on real data it over-predicts the rare classes. The **prior correction**
adds log(true class frequency) to the logits before argmax (Bayes prior shift). It needs no retraining and is one line of code.

| RoBERTa-base (v1) | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos | Neutral precision / recall |
|---|---|---|---|---|---|---|
| Balanced test | 0.827 | 0.827 | 0.810 | 0.748 | 0.925 | 0.74 / 0.75 |
| Natural test, raw | 0.889 | 0.762 | 0.850 | 0.483 | 0.953 | 0.35 / 0.76 |
| Natural test, prior-corrected | **0.924** | **0.781** | 0.871 | 0.499 | 0.971 | 0.51 / 0.49 |

Findings:
- On real data, accuracy is high (0.92) because most reviews are clearly positive.
- Neutral is the weak spot: only 6.4% of reviews, and genuinely ambiguous.
- Without correction, 65% of "neutral" predictions are wrong (precision 0.35). The prior correction trades neutral recall for precision and lifts accuracy by +3.5 points.

## RoBERTa-large (355M) vs RoBERTa-base (125M)
Colab T4, lr 1e-5, batch 8 x grad-accum 4 (effective 32), 2 epochs, ~2.2 h (one disconnect, resumed from the Drive checkpoint).

| Balanced test (15k) | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.780 | 0.780 | 0.776 | 0.690 | 0.874 |
| RoBERTa-base | 0.827 | 0.827 | 0.810 | 0.748 | 0.925 |
| **RoBERTa-large** | **0.833** | **0.833** | 0.815 | 0.755 | 0.930 |

| Natural test (20k, true mix) | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF | 0.842 | 0.703 |
| TF-IDF + prior | 0.897 | 0.694 |
| RoBERTa-base | 0.889 | 0.762 |
| RoBERTa-base + prior | 0.924 | 0.781 |
| RoBERTa-large | 0.894 | 0.768 |
| **RoBERTa-large + prior** | **0.927** | **0.789** |

Validation macro-F1 per half-epoch: 0.8278 → 0.8345 → 0.8368 → **0.8382** (still rising at the end, unlike base, which
plateaued after epoch 1). A third epoch would likely add a few tenths of a point.

**Cost/benefit (local inference, M2 fp16):**

| | Size | Speed | Balanced macro-F1 |
|---|---|---|---|
| RoBERTa-base | 249 MB | 21–27 reviews/s | 0.827 |
| RoBERTa-large | 711 MB | 7.8 reviews/s | 0.833 |

2.9x the size and 3x slower for +0.6 points. RoBERTa-base is the better deployment choice; large is the accuracy ceiling reference.

**Conclusion:** model size is no longer the bottleneck — label noise is. Both models sit at ~0.83, and their confident
errors are reviews whose star rating contradicts the text.

## Demo
`python app.py` starts a Gradio web demo at http://127.0.0.1:7860 (type a review, see class probabilities).


---

# v3 — 5 classes, one per star (RoBERTa-base)

Labels are now the star rating itself: 1★ strongly_negative · 2★ negative · 3★ neutral · 4★ positive · 5★ strongly_positive.
Data: 40,000 reviews per star → 160k train / 20k val / 20k test, plus a 20k natural-mix test set. Same model as v1
(roberta-base, 249 MB), lr 2e-5, batch 32, 2 epochs, ~1 h on a T4.

Exact-match accuracy is not the right yardstick for an ordered scale: predicting 4★ for a 5★ review is a small error,
predicting 1★ is a large one. So we also report **off-by-one accuracy**, **mean absolute error in stars** and
**quadratic weighted kappa** (QWK), the standard metric for ordinal ratings.

| Balanced test (20k) | Accuracy | Macro-F1 | Off-by-one | MAE (stars) | QWK |
|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.580 | 0.576 | 0.916 | 0.527 | 0.806 |
| **RoBERTa-base (5-class)** | **0.646** | **0.643** | **0.959** | **0.402** | **0.876** |

| Natural test (20k, true mix) | Accuracy | Macro-F1 | Off-by-one | MAE | QWK |
|---|---|---|---|---|---|
| TF-IDF | 0.702 | 0.541 | 0.932 | 0.401 | 0.824 |
| RoBERTa-base | 0.762 | 0.611 | 0.971 | 0.275 | 0.904 |
| **RoBERTa-base + prior** | **0.828** | 0.623 | 0.967 | **0.216** | **0.918** |

Per class (balanced test): 1★ F1 0.711 · 2★ 0.529 · 3★ 0.550 · 4★ 0.623 · 5★ 0.802.

Findings:
- Fine-tuning beats TF-IDF by +6.6 accuracy points and +7 QWK points on the same 5-class task.
- The model is rarely badly wrong: **96% of predictions land within one star**, average error 0.40 stars.
- Accuracy falls versus the 3-class task (0.646 vs 0.827) purely because the task is finer — 5 options, and the
  4★/5★ boundary is subjective even for people. QWK 0.876 is "very good agreement" on the usual scale.
- The extremes are easy (1★, 5★) and the middle is hard (2★, 3★), which is where star labels and review text
  disagree most — exactly what v4's label cleaning targets.

## Does the 5-class model replace the 3-class one?
Map its star probabilities down (1★+2★ → negative, 3★ → neutral, 4★+5★ → positive) and score it on the **original
3-class test set**, so it is directly comparable to v1. One correction is needed for fairness: the 5-class training
set is 40% negative / 20% neutral / 40% positive (two stars feed each end, one feeds the middle), while the test set
is a third each, so the training prior is divided out before the argmax.

| On the 3-class test set (15k) | Accuracy | Macro-F1 | F1 neu |
|---|---|---|---|
| v1, trained for 3 classes | 0.827 | 0.827 | 0.748 |
| v3 5-class model, mapped down | 0.796 | 0.781 | 0.631 |
| **v3 5-class model, mapped down + prior-corrected** | **0.823** | **0.820** | 0.723 |

**One model can serve both tasks**: within 0.8 points of the dedicated 3-class model while also predicting the exact
star. That halves what has to be deployed — relevant for a constrained environment. Without the prior correction the
same model loses 4.6 points, almost all of it neutral recall (0.50 vs 0.75), which shows the loss is a prior mismatch,
not a weaker model.

## Making it smaller: int8 quantization (no retraining)
Dynamic int8 quantization of the linear layers **and** the embedding table. Quantizing only the linear layers barely
helps, because the 50,265 x 768 word-embedding table is ~40% of the weights.

| `models/roberta_base_5class` (4,000 test reviews, CPU) | Size | Accuracy | Macro-F1 | QWK | Reviews/s |
|---|---|---|---|---|---|
| fp16 as trained | 249 MB | 0.647 | 0.644 | 0.883 | 19.2 |
| **int8 quantized** | **126 MB** | 0.645 | 0.642 | 0.881 | 5.6 |

**Half the size for 0.2 accuracy points.** The slowdown is specific to Apple Silicon: macOS PyTorch wheels ship only the
`qnnpack` backend, which is slower than Accelerate's fp32 path for these matrix sizes. On x86 Linux (`fbgemm`) int8
dynamic quantization is normally *faster* than fp32. The size saving holds on every platform.

Reproduce: `python quantize.py --model models/roberta_base_5class --limit 4000`.
