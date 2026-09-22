# v4 — Label-noise cleaning, then retrain

**Previous version:** [v3](../v3_star_level_5class) — 5-class model, accuracy 0.646, QWK 0.876, weakest on 2★/3★.

## Why
Star ratings are noisy labels. Real examples found in the training data:
- 1★ "These gloves work very well … I love these gloves"
- 5★ "One Star. It do not work?"
- 3★ "Three Stars. This is a really good product worth the money"

A model trained on contradictions learns worse. v1 and v2 showed that bigger models do not fix this,
so this version fixes the data instead.

## Method (`detect_label_noise.py`, confident learning with 2-fold cross-fitting)
1. Split the 160k training reviews into two halves.
2. Train on half A and predict half B; train on half B and predict half A.
   **Cross-fitting matters:** a model that trained on a review has partly memorised it, so its own prediction
   there cannot distinguish noise from memorisation.
3. Flag a review when the out-of-sample model is **confident** (p > 0.9) **and** its prediction is **at least 2
   classes away** from the label. A 4★ predicted as 5★ is ambiguity and is kept; a 1★ predicted as 5★ is a
   contradiction and is dropped.
4. Retrain roberta-base on what remains.

**The val and test sets are never cleaned**, so the comparison with v3 is fair. Cleaning the test set would
raise the score for a meaningless reason.

## Results
_Pending the Colab run. Fill in from `results/metrics.json` (keys `roberta_base_5class_cleaned*`)._

| Balanced test (20k) | Accuracy | Macro-F1 | Off-by-one | MAE | QWK |
|---|---|---|---|---|---|
| v3, trained on raw labels | 0.646 | 0.643 | 0.959 | 0.402 | 0.876 |
| v4, trained on cleaned labels | — | — | — | — | — |

Also produced: `results/label_noise.json` (how many reviews were removed, by star) and
`results/errors/label_noise_sample.csv` (the 100 clearest examples) — evidence that the dataset's quality was
measured, not just consumed.

## Files
`notebook.ipynb` · `results/` · put the downloaded zip in the root `outputs/`.
Reproduce: `python detect_label_noise.py --data-dir data5`, then `python finetune_roberta.py --name roberta_base_5class_cleaned --data-dir data5_clean --results-dir versions/v4_label_noise_cleaning/results`.
