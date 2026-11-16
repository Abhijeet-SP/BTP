# v5 — LoRA vs full fine-tuning

**Previous version:** [v4](../v4_label_noise_cleaning) — same model, cleaner training labels.

## Why
v1–v4 ask *how accurate?* This version asks **how cheaply that accuracy can be obtained**, which is the
constraint theme of the project. Same data, same hyper-parameters as v3; only the fine-tuning method changes.

**LoRA** freezes RoBERTa and trains small rank-8 matrices on the attention projections (query/key/value), plus the
classification head: **1,036,805 parameters, 0.82% of the model**. The optimizer then stores state for 1M values
instead of 125M, which is what allows big models to be tuned on small GPUs. The adapter alone is 4.2 MB.

**A common misconception:** LoRA does *not* shrink what you deploy. The adapter is a few MB, but it needs the full
base model to run, and merging it back gives the same 249 MB file. `finetune_roberta.py` merges it deliberately, so every
version ships in an identical format. What LoRA saves is *training* memory, not *inference* size.

## Evidence status
Training logs and figures exist, including the short memory-comparison runs. Final LoRA test scores and
peak-memory entries are absent from `results/metrics.json`; reconciliation is pending.
Do not treat the old “pending run” wording as proof that the models were never trained.

## Int8 deployment comparison
The [saved metrics](results/metrics.json) contain a 4,000-review balanced-test CPU comparison:

| Variant | Accuracy | Macro-F1 | QWK | Reviews/s |
|---|---:|---:|---:|---:|
| fp32 CPU | 0.6510 | 0.6473 | 0.8827 | 19.2 |
| int8 CPU | 0.6452 | 0.6418 | 0.8812 | 5.6 |

The JSON records 249 MB of stored source weights and a 126 MB int8 export. Runtime fp32 RAM is not
measured by the stored file size. The accuracy gap is 0.58 percentage points on this subset.
Int8 is slower in this comparison; the file does not include machine/backend metadata sufficient for
cross-platform speed conclusions. See [RESULTS.md](../../../RESULTS.md) for the evidence limits.

Reproduce in a separate results directory to preserve the original comparison:

```bash
python evaluate_model.py --model models/roberta_base_5class --data-dir datasets/sentiment_5class --quantize --results-dir before_mid_evaluation/local_results/int8_5class
```

## Files
`notebook.ipynb` · `results/` · this README. Put the downloaded zip in the root `downloads/colab_archives/`.

Run the commands above from the repository root (`main_btp/` locally, `/content/BTP` on Colab). This experiment is archived under `before_mid_evaluation/experiments/`; restructuring did not rerun it or change its saved results.
