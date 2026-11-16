# v2 — RoBERTa-large + real-world evaluation

**Previous version:** [v1](../v1_roberta_base_3class) — RoBERTa-base, 3 classes, macro-F1 0.827.

## What changed
1. **A 3x bigger model:** `roberta-large` (355M parameters), to test whether model size was the limit.
2. **A realistic test set:** `datasets/sentiment_3class/test_natural.csv`, 20k reviews in the true class mix (16.9% negative,
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
| RoBERTa-large | 0.894 | 0.768 |
| **RoBERTa-large + prior** | **0.927** | **0.789** |

The base-model natural-test scores from older notes are absent from the current v1 metric file; see [the evidence report](../../../RESULTS.md).

## What we learned
- The larger model added about 0.6 macro-F1 percentage points in this single-run comparison. RoBERTa-base remains the deployment architecture; fresh size/speed benchmarks and training variability checks are pending.
- Large's validation score was still rising at the end (0.828 → 0.838), unlike base, which had plateaued.
- Large-model natural-test accuracy changes from 0.8935 to 0.9272 with prior correction in the saved metrics.
- Confident star/text contradictions suggest label ambiguity, but no human-label study has established the noise rate or a performance ceiling.

**This motivated v3 and v4:** make the labels finer (v3) and cleaner (v4) rather than the model bigger.

## Files
`notebook.ipynb` · `results/`.

Run the commands above from the repository root (`main_btp/` locally, `/content/BTP` on Colab). This experiment is archived under `before_mid_evaluation/experiments/`; restructuring did not rerun it or change its saved results.
