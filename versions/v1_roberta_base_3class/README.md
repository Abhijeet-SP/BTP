# v1 — RoBERTa-base, 3-class sentiment

**Previous version:** none. This is the starting point.

## What this version does
Fine-tunes `roberta-base` (125M parameters, pretrained on ~160GB of text) to label an Amazon review as
negative / neutral / positive, and compares it against a classical TF-IDF + Logistic Regression baseline.

- Data: Amazon Reviews 2023, category `Industrial_and_Scientific` (5.18M reviews), from the Hugging Face mirror.
- Labels: 1–2★ negative, 3★ neutral, 4–5★ positive.
- Balanced sample: 50,000 per class via reservoir sampling (seed 42) → 120k train / 15k val / 15k test.
- Training: full fine-tuning, lr 2e-5, batch 32, 2 epochs, fp16, 42 min on a Colab T4.

## Results (balanced test set, 15k reviews)
| Model | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.780 | 0.780 | 0.776 | 0.690 | 0.874 |
| **RoBERTa-base** | **0.827** | **0.827** | 0.810 | 0.748 | 0.925 |

Fine-tuning gains **+4.7 points of macro-F1** over the baseline, most of it on the hard neutral class.

## What we learned
- Validation plateaued after epoch 1 (0.8286 → 0.8300), so more epochs would not help.
- 99% of errors are between adjacent classes; negative vs positive is confused only ~1% of the time.
- Error rate by star rating: 1★ 11%, 2★ 39%, 3★ 25%, 4★ 35%, 5★ 3.7% — errors cluster where the star rating and the
  review text disagree. The most confident mistakes are mislabelled reviews (see `results/errors/errors_roberta.csv`).
- Deployment: fp16 weights, 249 MB, ~27 reviews/s on an Apple M2 GPU. fp16 conversion cost zero accuracy.

**This motivated v2:** if the model is at 0.827, is the limit the model's size or the data?

## Files
`notebook.ipynb` the Colab run · `results/` this version's metrics, figures and logs.
Reproduce: `python prepare_data.py && python train_baseline_tfidf.py`, then the notebook on a Colab T4.
