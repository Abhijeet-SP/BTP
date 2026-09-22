# v7 — RoBERTa-base 3-class, int8 + prior correction

**Previous version:** [v6](../v6_qlora_4bit). **Model:** v1's `roberta_base_3class`, no new training.

## Why
v1–v6 show which choices pay off: full fine-tuning beats LoRA/QLoRA, label cleaning does not help, int8 halves the
size for ~0.2 points (v5), and the prior correction fixes the real-world class mix. v7 applies all of them to the
3-class model to get the smallest deployable sentiment model:
- **int8 dynamic quantization** of the Linear layers and the embedding table: 249 → ~126 MB.
- **Prior correction** on the natural mix: `(logits + np.log(prior / prior.sum())).argmax(-1)`. The model is trained
  on balanced classes, so it over-predicts negative/neutral on real data (16.9 / 6.4 / 76.7%); adding log(prior)
  moves only the borderline reviews. Nothing is retrained.

## Results
_Pending the Colab run._ fp32 and int8 are scored on the same 4,000 rows of the balanced and natural test sets.

## Run
Colab → `notebook.ipynb` → T4 GPU → Run all. It reuses the v1 zip on Drive
(`MyDrive/btp_v1_roberta_base_3class/`), or retrains v1 if it is missing. int8 inference runs on CPU.
Locally: `python evaluate_model.py --model models/roberta_base_3class --quantize --results-dir versions/v7_int8_3class_prior/results`

## Files
`notebook.ipynb` · `results/` · this README. Put the downloaded zip in the root `outputs/`.
