# v2 — RoBERTa-large + real-world evaluation

**Previous version:** [v1](../v1_roberta_base_3class) — RoBERTa-base, 3 classes, macro-F1 0.827.

## What changed
1. **A 3x bigger model:** `roberta-large` (355M parameters), to test whether model size was the limit.
2. **A realistic test set:** `data/test_natural.csv`, 20k reviews in the true class mix (16.9% negative,
   6.4% neutral, 76.7% positive), with no overlap with the balanced splits. The balanced test set makes rare
   classes look far more common than they are, which flatters the model.
3. **A prior correction:** models trained on balanced classes over-predict rare classes on real data. Adding
   log(true class frequency) to the logits fixes this in one line, with no retraining.
4. **A Gradio demo** (`python predict_sentiment.py --web`).

The balanced splits were kept byte-identical to v1 (md5-verified), so the comparison is exact.

## Results
| Balanced test (15k) | Accuracy | Macro-F1 |
|---|---|---|
| RoBERTa-base (v1) | 0.827 | 0.827 |
| **RoBERTa-large** | **0.833** | **0.833** |

| Natural test (20k, true mix) | Accuracy | Macro-F1 |
|---|---|---|
| RoBERTa-base | 0.889 | 0.762 |
| RoBERTa-base + prior | 0.924 | 0.781 |
| RoBERTa-large | 0.894 | 0.768 |
| **RoBERTa-large + prior** | **0.927** | **0.789** |

## What we learned
- **Tripling the model bought +0.6 points.** Cost: 711 MB vs 249 MB, and 7.8 vs 27 reviews/s on an M2.
  For a constrained deployment, base is the right choice; large is the accuracy ceiling reference.
- Large's validation score was still rising at the end (0.828 → 0.838), unlike base, which had plateaued.
- **The prior correction is nearly free and worth +3.5 accuracy points** on real-world data.
- Conclusion: the bottleneck is **label noise, not model capacity**. Both models sit near 0.83 and their
  confident errors are reviews whose star rating contradicts the text.

**This motivated v3 and v4:** make the labels finer (v3) and cleaner (v4) rather than the model bigger.

## Files
`notebook.ipynb` · `results/`.
