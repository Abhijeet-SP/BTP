# v6 — QLoRA: LoRA on a 4-bit frozen base

**Previous version:** [v5](../v5_lora_finetuning) — LoRA trains ~1.5% of the weights at full precision.

## What changed
QLoRA goes one step further: the base model is loaded in **4-bit NF4** and frozen, and only the LoRA matrices train,
in fp16. Memory during training drops again, because the frozen weights occupy a quarter of the space.

| | Base weights | Trained weights | Made for |
|---|---|---|---|
| Full fine-tuning (v3) | fp16, updated | 124.6M (100%) | plenty of GPU |
| LoRA (v5) | fp16, frozen | 1.04M (0.82%) | medium GPU |
| **QLoRA (v6)** | **4-bit, frozen** | 1.04M (0.82%) | small GPU / very large models |

## Evidence status
Training logs and confusion/loss figures exist in `results/`. Final QLoRA test metrics and training-memory
measurements have not been reconciled: `results/metrics.json` is empty. Report those measurements as
pending rather than assuming that no training occurred or quoting unlinked numbers.

The experiment compares accuracy, memory and training time with full fine-tuning and LoRA.
Use recorded runs to establish these trade-offs; the architecture alone does not establish their size.
See [RESULTS.md](../../../RESULTS.md) for the current evidence status.

The saved model is a normal fp16 RoBERTa in every case: the adapter is merged into a freshly loaded base, so all
versions deploy identically (~249 MB) and the comparison stays apples-to-apples.

## Files
`notebook.ipynb` · `results/` · this README. Needs `bitsandbytes`, so it runs on a CUDA GPU (Colab), not on a Mac.

Run the commands above from the repository root (`main_btp/` locally, `/content/BTP` on Colab). This experiment is archived under `before_mid_evaluation/experiments/`; restructuring did not rerun it or change its saved results.
