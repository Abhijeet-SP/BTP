# v7 — Three-class int8 comparison; prior evaluation pending

**Previous version:** [v6](../v6_qlora_4bit). Uses v1's `roberta_base_3class`.
The folder name preserves the original experiment intent: quantization plus natural-prior evaluation.

## Current supported workflow
`notebook.ipynb` obtains the v1 model from Drive (or uses the v1 training recipe if no saved model is available),
then calls `evaluate_model.py --quantize`. The evaluator compares fp32 CPU and int8 CPU on the same
4,000 balanced-test rows, saves the int8 state dict, and writes a nested comparison under
`roberta_base_3class_quantized` in `results/metrics.json`.

The notebook summary now reads that actual schema. Both variants run on CPU; the comparison is not
fp32 GPU versus int8 CPU. Quantized natural-test and prior-corrected evaluation are not implemented by
this command and remain pending. No v7 metrics file is currently saved in the repository.

## Run
Open the notebook on Colab, or use the existing local model:

```bash
python evaluate_model.py --model models/roberta_base_3class --data-dir datasets/sentiment_3class --quantize --results-dir before_mid_evaluation/local_results/int8_3class
```

Use a separate results destination for repeat runs. The notebook exports its model/results ZIP;
keep downloads in `downloads/colab_archives/`. Run Python commands from the repository root.
The [evidence report](../../../RESULTS.md) distinguishes current saved results from pending work.
