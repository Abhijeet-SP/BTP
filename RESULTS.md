# Results and evidence

This report summarises existing saved measurements. Documentation was reconciled during the folder/path audit;
no training or full evaluation was rerun. Original metric files, logs, figures, datasets and models are unchanged.
`README.md` explains how to run the project; this file explains what the saved evidence supports.

## Three-class sentiment

Labels: 1–2 stars = negative, 3 = neutral, 4–5 = positive.
The balanced test contains 15,000 reviews; the natural test contains 20,000.

| Balanced test | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.7802 | 0.7802 |
| RoBERTa-base | 0.8270 | 0.8273 |
| RoBERTa-large | 0.8333 | 0.8334 |

Sources: [v1 metrics](before_mid_evaluation/experiments/v1_roberta_base_3class/results/metrics.json),
[v2 metrics](before_mid_evaluation/experiments/v2_roberta_large_3class/results/metrics.json).
Base improves macro-F1 over TF-IDF; large adds about 0.6 percentage points over base in these single runs.
The larger model's benefit must be weighed against deployment cost. Statistical significance has not been established.

| Natural test | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF | 0.8417 | 0.7030 |
| TF-IDF + prior correction | 0.8972 | 0.6943 |
| RoBERTa-large | 0.8935 | 0.7682 |
| RoBERTa-large + prior correction | 0.9272 | 0.7889 |

These values come from the same v1/v2 metric files. Earlier notes reported natural-test base-model scores,
but those entries are absent from the current v1 metric file, so they are not included as reconciled results here.
Prior correction adds log(natural class frequency) to logits; it changes predictions without retraining.
It requires priors appropriate to the target population.

## Five-class sentiment: the main task

Each star is a class. The balanced test contains 20,000 reviews; the saved natural-test reports contain 19,999.
MAE measures error in stars; off-by-one accuracy accepts a prediction within one star; QWK is quadratic weighted kappa.

| Balanced test | Accuracy | Macro-F1 | Off-by-one | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.5797 | 0.5764 | 0.9158 | 0.5273 | 0.8064 |
| RoBERTa-base | 0.6461 | 0.6427 | 0.9588 | 0.4016 | 0.8759 |

Source: [v3 metrics](before_mid_evaluation/experiments/v3_star_level_5class/results/metrics.json).
RoBERTa-base beats the baseline on these saved test metrics. Its exact-star accuracy is lower than the
three-class task's accuracy; the tasks have different labels and should not be ranked by that number alone.

| Natural test | Accuracy | Macro-F1 | Off-by-one | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.7022 | 0.5409 | 0.9317 | 0.4011 | 0.8242 |
| RoBERTa-base | 0.7622 | 0.6113 | 0.9712 | 0.2753 | 0.9043 |
| RoBERTa-base + prior correction | 0.8281 | 0.6225 | 0.9666 | 0.2165 | 0.9179 |

The same v3 file also records the five-class model mapped to the original three-class test:

| Five-class model mapped to three classes | Accuracy | Macro-F1 |
|---|---:|---:|
| Mapped probabilities | 0.7955 | 0.7814 |
| Mapped + prior correction | 0.8232 | 0.8197 |

## Label-cleaning experiment

[v4 cleaning summary](before_mid_evaluation/experiments/v4_label_noise_cleaning/results/label_noise.json)
records 145 removed rows out of 160,000 training rows (0.09%), using probability > 0.9 and a class distance
of at least two. Validation and test data are not cleaned.

The [v4 model metric file](before_mid_evaluation/experiments/v4_label_noise_cleaning/results/metrics.json)
is empty. Figures and training logs exist, but final cleaned-model test scores have not been reconciled.
The small flagged fraction describes this detector at these thresholds; it does not prove the remaining
labels are correct or that other cleaning methods cannot help.

## LoRA, QLoRA and int8

LoRA/QLoRA target training memory. Their saved adapters are merged into an fp16 base for deployment,
so they do not by themselves shrink the deployed model. Int8 is the deployment-size experiment.

The [v5 metric file](before_mid_evaluation/experiments/v5_lora_finetuning/results/metrics.json)
contains one historical int8 comparison on 4,000 balanced-test reviews:

| CPU inference variant | Accuracy | Macro-F1 | MAE | QWK | Reviews/s |
|---|---:|---:|---:|---:|---:|
| fp32 CPU | 0.6510 | 0.6473 | 0.3912 | 0.8827 | 19.2 |
| int8 CPU | 0.6452 | 0.6418 | 0.3967 | 0.8812 | 5.6 |

The file records 249 MB of stored source weights and a 126 MB int8 export.
The source file size does not describe fp32 runtime RAM. The recorded accuracy difference is
0.58 percentage points on this subset.
The JSON labels the reference **fp32 CPU**, not fp16. It does not record sufficient machine/backend
metadata to establish a cross-platform speed claim. Int8 is slower in this saved comparison;
x86 speed and peak RAM still need measurement. The old “0.2-point drop” wording is not supported by these rows.

Final LoRA test metrics are absent from v5; [v6 metrics](before_mid_evaluation/experiments/v6_qlora_4bit/results/metrics.json)
are empty. Existing logs and figures are retained. Do not present their reported final scores as reconciled evidence yet.
The v7 metric file is absent. Its corrected notebook can run the existing balanced CPU comparison;
quantized natural/prior evaluation remains pending.

## Reproduction and limitations

Use [the project README](README.md) and [notebook index](before_mid_evaluation/README.md).
For local predictions: `python predict_sentiment.py "review text"`; for the Gradio demo:
`python predict_sentiment.py --web`. Models and datasets are local/ignored and must be supplied or generated
on a fresh checkout. No after-mid implementation has been added.

- Existing comparisons use single training runs. Additional seeds are required to quantify training variability.
- Labels are derived from stars. Confident contradictions suggest possible label noise; no human-labelled
  study has established its rate or a performance ceiling.
- The data comes from one Amazon category. Materials-specific and external-domain performance is unmeasured.
- Training-time fields for some resumed experiments are very short; do not treat them as complete
  end-to-end training durations without reconciling logs/checkpoints.
- Corrected small-subset evaluation includes all model classes in macro-F1 and reports undefined QWK as null.
  Historical full-test results are unchanged.
- Exported Word/PDF files under `before_mid_evaluation/reports/` are historical exports and may contain
  older wording. This Markdown report records the current evidence status.

Missing evidence, CPU benchmarks, additional seeds, human labelling, the offline CSV tool, packaging,
and final report/slides remain in the separately authorized workspace plan. `after_mid_evaluation/` stays empty.
