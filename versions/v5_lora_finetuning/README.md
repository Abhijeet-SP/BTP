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

## Results
_Pending the Colab run (`notebook.ipynb`). The notebook also runs a 100-step benchmark of both methods to record
peak GPU memory._

| | Trainable parameters | Peak GPU | Accuracy | QWK |
|---|---|---|---|---|
| Full fine-tuning (v3) | 124,649,477 (100%) | 5.11 GB | 0.646 | 0.876 |
| LoRA rank 8 | 1,036,805 (0.82%) | — | — | — |

## Related: making the model smaller (already done)
Deployment size is cut by **quantization**, not by LoRA. Dynamic int8 on the linear layers *and* the embedding
table (the 50,265 x 768 word table is ~40% of the weights):

| `models/roberta_base_5class` (4,000 test reviews, CPU) | Size | Accuracy | Macro-F1 | QWK | Reviews/s |
|---|---|---|---|---|---|
| fp16 as trained | 249 MB | 0.651 | 0.647 | 0.883 | 19.2 |
| **int8 quantized** | **126 MB** | 0.645 | 0.642 | 0.881 | 5.6 |

Half the size for 0.2 accuracy points. The slowdown is specific to Apple Silicon, whose PyTorch build ships only the
`qnnpack` backend; on x86 Linux (`fbgemm`) int8 is normally faster. The size saving holds everywhere.

Reproduce: `python evaluate_model.py --model models/roberta_base_5class --data-dir data5 --quantize --results-dir versions/v5_lora_finetuning/results`

## Files
`notebook.ipynb` · `results/` · this README. Put the downloaded zip in the root `outputs/`.
