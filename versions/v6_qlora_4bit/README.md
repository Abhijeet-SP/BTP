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

## Honest expectation
QLoRA was designed to fine-tune models with *billions* of parameters on one consumer GPU. RoBERTa-base has 125M, and
full fine-tuning already fits in 5.11 GB of a 16 GB T4 (measured), so there is no memory problem here to solve. What this
version contributes is the **measurement**: how accuracy, memory and speed move when the base is quantized to 4 bits.
Expect similar accuracy, lower peak memory, and *slower* steps, because 4-bit weights are dequantized on the fly.
That trade-off is the result worth reporting, and it completes the constrained fine-tuning story:
full fine-tuning → LoRA → QLoRA → int8 deployment.

## Results
_Pending the Colab run (`notebook.ipynb`)._

| | Trainable parameters | Peak GPU | Training time | Accuracy | QWK |
|---|---|---|---|---|---|
| Full fine-tuning (v3) | 124,649,477 | 5.11 GB | ~60 min | 0.646 | 0.876 |
| LoRA (v5) | 1,036,805 | — | — | — | — |
| QLoRA (v6) | 1,036,805 | — | — | — | — |

The saved model is a normal fp16 RoBERTa in every case: the adapter is merged into a freshly loaded base, so all
versions deploy identically (~249 MB) and the comparison stays apples-to-apples.

## Files
`notebook.ipynb` · `results/` · this README. Needs `bitsandbytes`, so it runs on a CUDA GPU (Colab), not on a Mac.
