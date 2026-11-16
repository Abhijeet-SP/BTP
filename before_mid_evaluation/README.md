# Before mid evaluation — existing work

This folder holds the existing v1–v7 experiment collection and exported reports. “Before mid evaluation”
identifies the project phase; it does not mean every experiment has complete evidence.

## Experiment navigation

| Folder | Task | Notebook |
|---|---|---|
| [v1](experiments/v1_roberta_base_3class/) | Three-class base model and classical baseline | [Open](experiments/v1_roberta_base_3class/notebook.ipynb) |
| [v2](experiments/v2_roberta_large_3class/) | Three-class large model | [Open](experiments/v2_roberta_large_3class/notebook.ipynb) |
| [v3](experiments/v3_star_level_5class/) | Five-class base model and classical baseline | [Open](experiments/v3_star_level_5class/notebook.ipynb) |
| [v4](experiments/v4_label_noise_cleaning/) | Training-label cleaning | [Open](experiments/v4_label_noise_cleaning/notebook.ipynb) |
| [v5](experiments/v5_lora_finetuning/) | LoRA, training memory and int8 comparison | [Open](experiments/v5_lora_finetuning/notebook.ipynb) |
| [v6](experiments/v6_qlora_4bit/) | QLoRA | [Open](experiments/v6_qlora_4bit/notebook.ipynb) |
| [v7](experiments/v7_int8_3class_prior/) | Three-class int8/prior experiment | [Open](experiments/v7_int8_3class_prior/notebook.ipynb) |

Each experiment contains its README, notebook and `results/` directory. Results include metrics,
figures, training logs and error samples where available. Existing report exports are in [reports/](reports/).
The evidence summary is [RESULTS.md](../RESULTS.md); existing Word/PDF reports remain historical exports.

## How the pieces connect

Run all Python commands from the repository root, not from inside this folder.
Notebooks clone the GitHub repository into `/content/BTP` on Colab and run the root scripts there.
They prepare a task dataset, train/evaluate its model, save checkpoints on Google Drive, and export a ZIP.
The same shared scripts support both class counts; there are no duplicated training-script copies per task.

- Original data: `datasets/raw_reviews/`.
- Prepared tasks: `datasets/sentiment_3class/` and `datasets/sentiment_5class/`.
- Data files: `train.csv`, `validation.csv`, `test_balanced.csv`, `test_natural.csv`, `dataset_stats.json`.
- Saved usable models: `models/`.
- Downloaded ZIP packages: `downloads/colab_archives/`.
- Experiment evidence: `before_mid_evaluation/experiments/<version>/results/`.

The filenames explain their purpose: training learns weights; validation checks training progress;
balanced testing compares classes fairly; natural testing measures the real-world rating mix.

## Preservation and known gaps

Restructuring moves the existing files and updates paths. It does not regenerate datasets, rerun training,
replace result values, or implement after-mid tasks. Original model names, metric keys, training settings,
Drive/checkpoint paths and downloaded archives are retained.

v4/v6 metric files are empty, v5 has only the int8 comparison entry, and v7 has no metric file.
These are pre-existing gaps; historical report claims remain unverified where supporting evidence is absent.
Full notebook execution is not certified by local checks. v7's summary matches the existing balanced
CPU comparison; its quantized natural/prior evaluation remains pending.
The separate work plan records the evidence-reconciliation tasks for later authorization.
