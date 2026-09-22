# Fine-tuning RoBERTa for Review Sentiment under Constraints
**B.Tech Project · Amazon Reviews 2023 — Industrial & Scientific (5.18M reviews)**

How good can review-sentiment prediction get with a model small enough to run on a laptop?
Each version changes exactly one thing — model size, label granularity, label quality, or fine-tuning method —
so every result is attributable.

| Version | Question | Result |
|---|---|---|
| [v1](versions/v1_roberta_base_3class) | RoBERTa-base vs TF-IDF, 3 classes | macro-F1 **0.827** vs 0.780 |
| [v2](versions/v2_roberta_large) | Does a 3x bigger model help? | 0.833 (+0.6) for 2.9x the size — **no** |
| [v3](versions/v3_star_level_5class) | 5 classes, one per star (intensity) | acc 0.646, **96% within one star**, QWK 0.876 |
| [v4](versions/v4_label_noise_cleaning) | Remove mislabelled reviews, retrain | _running_ |
| [v5](versions/v5_lora_finetuning) | LoRA vs full fine-tuning; int8 size | int8: **249 → 126 MB**, −0.2 points |
| [v6](versions/v6_qlora_4bit) | QLoRA: LoRA on a 4-bit frozen base | _planned_ |

Full analysis and every table: **[RESULTS.md](RESULTS.md)**. Each version folder holds its own README
(what changed and why), its Colab notebook, and its results.

## Layout
```
prepare_data.py  finetune_roberta.py  train_baseline_tfidf.py  detect_label_noise.py  evaluate_model.py  predict_sentiment.py   shared code, used by every version
versions/<version>/  README.md · notebook.ipynb · results/{metrics.json, figures/, logs/, errors/}
outputs/             downloaded result zips (not in git)
data/ data5/ models/ generated locally (not in git)
```

## Data
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), category `Industrial_and_Scientific`, via the
Hugging Face mirror. Text = `title + ". " + body`; reviews under 3 words and exact duplicates removed.
Balanced samples are drawn by reservoir sampling with a fixed seed, so `prepare_data.py` reproduces the same splits
anywhere. A separate test set preserves the real class mix (77% positive) for realistic evaluation.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

python prepare_data.py                                  # 3-class splits (downloads 2.3GB once)
python prepare_data.py --classes 5 --per-class 40000    # 5-class splits
python train_baseline_tfidf.py                              # TF-IDF comparison
python predict_sentiment.py "Stopped working after a week"   # try the model
python predict_sentiment.py --web                            # browser demo
python evaluate_model.py --model models/roberta_base_5class --data-dir data5 --quantize
```
Training runs on a free Colab T4: open the `notebook.ipynb` inside a version folder and Run all.
Inference runs locally (~27 reviews/s in fp16 on an Apple M2).

## How models are judged
Balanced test sets for macro-F1; a natural-mix test set for realistic accuracy; a Bayes prior correction when a
balanced-trained model meets real-world data. For the star-level task, ordered-label metrics too: off-by-one
accuracy, mean absolute error in stars, and quadratic weighted kappa.
