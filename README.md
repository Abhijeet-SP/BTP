# Fine-tuning RoBERTa for Private, Local Review Analysis

B.Tech project · Amazon Reviews 2023 · Industrial & Scientific.
The aim is a sentiment model small enough for local analysis of feedback, with measured trade-offs
between prediction quality, model size, speed and memory. Existing experiments are preserved below;
upcoming work is planned separately in the MBTP workspace's `agents_info/work.md`.

## Folder map

```text
main_btp/
├── datasets/
│   ├── raw_reviews/                 original downloaded Amazon reviews
│   ├── sentiment_3class/            three-class prepared splits
│   └── sentiment_5class/            five-class prepared splits
├── before_mid_evaluation/
│   ├── README.md                    workflow and experiment navigation
│   ├── experiments/<version>/      README, notebook, and results
│   └── reports/                     existing Word/PDF reports
├── after_mid_evaluation/            empty local folder; no work implemented
├── models/                          actual trained models and int8 files
├── downloads/colab_archives/         original downloaded Colab ZIPs
├── prepare_data.py
├── train_baseline_tfidf.py
├── finetune_roberta.py
├── detect_label_noise.py
├── evaluate_model.py
├── predict_sentiment.py
├── RESULTS.md
└── requirements.txt
```

Datasets, models, downloads, `.venv/`, and the reserved after-mid folder are ignored by Git.
Model files and archives are real directories inside this project, with no external model/output links.
An empty folder is not included in a Git clone. Create `after_mid_evaluation/` locally if needed;
its ignore rule can be removed when the owner authorizes work there.

## Existing experiments

| Version | Question |
|---|---|
| [v1](before_mid_evaluation/experiments/v1_roberta_base_3class/) | Three-class RoBERTa-base versus TF-IDF + Logistic Regression |
| [v2](before_mid_evaluation/experiments/v2_roberta_large_3class/) | Does RoBERTa-large justify its size and inference cost? |
| [v3](before_mid_evaluation/experiments/v3_star_level_5class/) | Five ordered sentiment classes, one per star |
| [v4](before_mid_evaluation/experiments/v4_label_noise_cleaning/) | Detect suspected training-label noise, remove flagged rows, retrain |
| [v5](before_mid_evaluation/experiments/v5_lora_finetuning/) | LoRA versus full fine-tuning; int8 deployment comparison |
| [v6](before_mid_evaluation/experiments/v6_qlora_4bit/) | QLoRA on a frozen 4-bit base |
| [v7](before_mid_evaluation/experiments/v7_int8_3class_prior/) | Three-class int8 and prior-correction experiment |

[Before-mid workflow](before_mid_evaluation/README.md) explains the steps and artifacts.
[RESULTS.md](RESULTS.md) summarises saved measurements with source links and explicit evidence gaps.
It is separate from this setup guide. Historical Word/PDF exports remain unchanged in the reports folder.

## Workflow

1. `prepare_data.py` downloads the source once into `datasets/raw_reviews/`, cleans text,
   constructs labels, samples reviews and writes the prepared splits.
2. `train_baseline_tfidf.py` trains the classical comparison model on the selected task.
3. Each experiment's `notebook.ipynb` runs shared scripts on Colab. `finetune_roberta.py`
   adapts pretrained RoBERTa; Drive checkpoints allow training to resume.
4. Shared evaluation writes metrics, figures, logs and error samples into that experiment's `results/`.
5. Colab archives are downloaded into `downloads/colab_archives/`. Place model files in `models/`
   and result evidence in the matching experiment directory; preserve existing files when importing runs.
6. `predict_sentiment.py` uses a saved model locally; `evaluate_model.py` supports local evaluation and int8 comparisons.

Historical ZIPs retain their original internal paths. Extract older packages into a staging folder,
then place their models and result evidence into the current matching directories.

## Datasets

Both tasks share the same source. Three-class labels group 1–2 stars as negative, 3 as neutral,
and 4–5 as positive. Five-class labels correspond to individual stars.

| Dataset | Training | Validation | Balanced test | Natural test |
|---|---:|---:|---:|---:|
| `datasets/sentiment_3class/` | 120,000 | 15,000 | 15,000 | 20,000 |
| `datasets/sentiment_5class/` | 160,000 | 20,000 | 20,000 | 19,999 |

Each contains `train.csv`, `validation.csv`, `test_balanced.csv`, `test_natural.csv`, and
`dataset_stats.json`. CSVs contain `text,label,rating`. Existing metadata retains its original
split keys (`train`, `val`, `test`, `test_natural`); only the filenames changed.
The five-class cleaning workflow creates `datasets/sentiment_5class_cleaned/` when run;
that dataset is not currently present locally. Validation and test files remain unchanged by cleaning.

## Run from the repository root

Use the existing environment locally:

```bash
source .venv/bin/activate
python predict_sentiment.py "Stopped working after a week"
python predict_sentiment.py --web
```

For a fresh checkout, set up the environment first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The code targets Transformers 5; the requirement constrains that major version. QLoRA additionally
needs `bitsandbytes` on CUDA; its notebook installs it separately. Fresh checkouts have no model weights:
copy an existing complete model folder into `models/` or obtain one through the appropriate Colab notebook
before running predictions. The downloadable model package is future work.

Full training runs on Colab GPU, not the Mac. Open the relevant experiment notebook. The owner must commit/push the
restructured scripts before those notebooks can clone this layout from GitHub.

Only generate datasets on a fresh setup; existing splits do not need regeneration after a move:

```bash
python prepare_data.py
python prepare_data.py --classes 5 --per-class 40000
```

Use an explicit dataset, model and results destination for evaluation. This example writes to v3;
use a separate destination for a repeat run to preserve historical evidence. When `--results-dir`
is omitted, scripts use `before_mid_evaluation/local_results/` so they do not overwrite archived
experiment metrics by default (that folder is created only when a run writes results):

```bash
python evaluate_model.py --model models/roberta_base_5class --data-dir datasets/sentiment_5class --results-dir before_mid_evaluation/experiments/v3_star_level_5class/results
```

Balanced-test macro-F1 is the primary comparison metric. For five ordered classes also report
accuracy, off-by-one accuracy, mean absolute error and quadratic weighted kappa. Natural-distribution
and prior-corrected scores are separate measurements. Original scientific settings and saved artifacts
were preserved during restructuring; full training and notebook execution were not repeated.

## What belongs on GitHub

Include shared scripts, requirements, READMEs, `RESULTS.md`, experiment notebooks, saved experiment
metrics/figures/logs/errors, and the existing report exports. `.gitignore` excludes datasets, model weights,
downloaded ZIPs, the environment, generated caches, scratch `before_mid_evaluation/local_results/`, and
the reserved after-mid folder. Scratch results must be reviewed before being adopted as experiment evidence.

The old `versions/` paths appear as removed files in Git; their replacements are under
`before_mid_evaluation/experiments/`. Include both sides of these moves when preparing your changes.
The sibling `agents_info/` folder and workspace guide are local workspace material, outside this repository.
