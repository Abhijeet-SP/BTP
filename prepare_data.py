"""Download Amazon Reviews 2023 (Industrial_and_Scientific), build balanced train/val/test CSVs.

Usage: python prepare_data.py [--per-class 50000] [--natural 20000]          # 3 classes -> data/
       python prepare_data.py --classes 5 --per-class 40000                  # 5 classes (one per star) -> data5/

Also writes data/test_natural.csv: a test set with the real-world class mix (~77% positive).
Per-class uniform samples (own RNG, so the balanced splits stay identical to v1), excluding every
review used in the balanced splits, with class counts set to the true class proportions.
"""
import argparse, csv, json, random, re, statistics
import urllib.request
from collections import Counter
from pathlib import Path

# Hugging Face mirror of https://amazon-reviews-2023.github.io/ (UCSD server was ~30x slower)
URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Industrial_and_Scientific.jsonl"
RAW = Path("data/raw/Industrial_and_Scientific.jsonl")
LABELS = ["negative", "neutral", "positive"]
LABELS_5 = ["strongly_negative", "negative", "neutral", "positive", "strongly_positive"]  # 1..5 stars


def label_names(n):
    return LABELS if n == 3 else LABELS_5


def data_dir(classes):
    return "data" if classes == 3 else f"data{classes}"


def stats_path(dir_):
    """Dataset stats live with the dataset, so any version can read the class priors."""
    return f"{dir_}/stats.json"


def default_model():
    """Newest trained model available locally, best first: cleaned 5-class, 5-class, large, base."""
    for m in ("roberta_base_5class_cleaned", "roberta_base_5class_qlora", "roberta_base_5class_lora", "roberta_base_5class", "roberta_large_3class", "roberta_base_3class"):
        if Path(f"models/{m}/config.json").exists():
            return f"models/{m}"
    raise SystemExit("No trained model in models/ — download one from a version's outputs zip.")


def rating_to_label(r, classes=3):
    if classes == 5:
        return r - 1  # 1 star -> 0 ... 5 stars -> 4
    return 0 if r <= 2 else 1 if r == 3 else 2


def clean(title, text):
    s = f"{title.strip()}. {text.strip()}" if title.strip() else text.strip()
    s = re.sub(r"<br\s*/?>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def save_results(name, y_true, y_pred, extra=None, results_dir="results"):
    """Shared by baseline/train/evaluate: merge metrics into results/metrics.json, save confusion matrix PNG.

    Classes are inferred from the labels, so this works for the 3-class and the 5-class task.
    For ordered labels (stars) the distance matters, so we also report mean absolute error in classes,
    off-by-one accuracy, and quadratic weighted kappa (the standard metric for ordinal ratings).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                                 cohen_kappa_score, f1_score)

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    names = label_names(max(int(y_true.max()), int(y_pred.max())) + 1)
    rdir = Path(results_dir)
    (rdir / "figures").mkdir(parents=True, exist_ok=True)
    out = rdir / "metrics.json"
    metrics = json.loads(out.read_text()) if out.exists() else {}
    metrics[name] = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "mae_classes": round(float(np.abs(y_true - y_pred).mean()), 4),
        "off_by_one_accuracy": round(float((np.abs(y_true - y_pred) <= 1).mean()), 4),
        "quadratic_weighted_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 4),
        "report": classification_report(y_true, y_pred, target_names=names, output_dict=True, digits=4, zero_division=0),
        **(extra or {}),
    }
    out.write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0))
    print(f"{name}: MAE {metrics[name]['mae_classes']}  off-by-one {metrics[name]['off_by_one_accuracy']}  "
          f"QWK {metrics[name]['quadratic_weighted_kappa']}")

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=names, normalize="true",
                                            values_format=".2f", cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} — test set (row-normalized)")
    plt.tight_layout()
    plt.savefig(rdir / "figures" / f"confusion_matrix_{name}.png", dpi=150)
    plt.close()


def evaluate_all(name, predict_logits, extra=None, limit=0, dir_="data", results_dir="results"):
    """Score a model on <dir_>/test.csv and (if present) <dir_>/test_natural.csv.

    predict_logits(texts) -> array [n, classes] of logits or log-probabilities.
    The model is trained on balanced classes, so on the natural mix it over-predicts the rare classes.
    `<name>_natural_prior` fixes that by adding log(natural prior) to the logits (Bayes prior shift;
    the uniform training prior cancels). Returns the test dataframe and its logits.
    """
    import numpy as np
    import pandas as pd

    test = pd.read_csv(f"{dir_}/test.csv", nrows=limit or None)
    logits = np.asarray(predict_logits(test.text.tolist()))
    save_results(name, test.label, logits.argmax(-1), extra, results_dir)
    nat_file = Path(f"{dir_}/test_natural.csv")
    if nat_file.exists():
        nat = pd.read_csv(nat_file, nrows=limit or None)
        nl = np.asarray(predict_logits(nat.text.tolist()))
        save_results(f"{name}_natural", nat.label, nl.argmax(-1), results_dir=results_dir)
        valid = json.loads(Path(stats_path(dir_)).read_text())["valid_per_class"]
        prior = np.array([valid[l] for l in label_names(len(valid))], dtype=float)
        save_results(f"{name}_natural_prior", nat.label, (nl + np.log(prior / prior.sum())).argmax(-1),
                     results_dir=results_dir)
    return test, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=3, choices=[3, 5])
    ap.add_argument("--per-class", type=int, default=50000)  # matches the Colab run
    ap.add_argument("--natural", type=int, default=20000, help="size of natural-distribution test set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    C = args.classes
    out_dir = data_dir(C)
    Path(out_dir).mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    nat_rng = random.Random(args.seed + 1)  # separate stream: balanced splits unchanged from v1

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, RAW)

    k = args.per_class
    reservoirs = {c: [] for c in range(C)}
    natural = {c: [] for c in range(C)}  # per-class reservoirs; trimmed to the true mix at the end
    seen = Counter()          # unique, valid reviews per class (reservoir counter)
    rating_counts = Counter() # raw rating distribution, for the report
    hashes = set()
    total = dropped = 0
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            total += 1
            r = json.loads(line)
            rating = int(r["rating"])
            rating_counts[rating] += 1
            text = clean(r.get("title", ""), r.get("text", ""))
            h = hash(text)
            if len(text.split()) < 3 or h in hashes:
                dropped += 1
                continue
            hashes.add(h)
            c = rating_to_label(rating, C)
            seen[c] += 1
            row = (text, c, rating)
            if len(reservoirs[c]) < k:
                reservoirs[c].append(row)
            else:
                j = rng.randrange(seen[c])
                if j < k:
                    reservoirs[c][j] = row
            if len(natural[c]) < args.natural:
                natural[c].append(row)
            else:
                j = nat_rng.randrange(seen[c])
                if j < args.natural:
                    natural[c][j] = row
            if total % 500000 == 0:
                print(f"  read {total:,} lines")

    splits = {"train": [], "val": [], "test": []}
    for c, rows in reservoirs.items():
        rng.shuffle(rows)
        n = len(rows)
        a, b = int(0.8 * n), int(0.9 * n)
        splits["train"] += rows[:a]
        splits["val"] += rows[a:b]
        splits["test"] += rows[b:]
    balanced = {t for rows in reservoirs.values() for t, _, _ in rows}
    splits["test_natural"] = []
    for c, rows in natural.items():
        rows = [r for r in rows if r[0] not in balanced]
        need = round(args.natural * seen[c] / sum(seen.values()))
        assert len(rows) >= need, f"not enough {label_names(C)[c]} reviews for natural test"
        nat_rng.shuffle(rows)
        splits["test_natural"] += rows[:need]
    for name, rows in splits.items():
        (nat_rng if name == "test_natural" else rng).shuffle(rows)
        with open(f"{out_dir}/{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label", "rating"])
            w.writerows(rows)

    lengths = [len(t.split()) for rows in reservoirs.values() for t, _, _ in rows]
    stats = {
        "total_reviews": total,
        "dropped_short_or_duplicate": dropped,
        "rating_distribution": dict(sorted(rating_counts.items())),
        "classes": C,
        "valid_per_class": {label_names(C)[c]: seen[c] for c in range(C)},
        "sampled_per_class": {label_names(C)[c]: len(reservoirs[c]) for c in range(C)},
        "split_sizes": {k_: len(v) for k_, v in splits.items()},
        "test_natural_per_class": {label_names(C)[c]: sum(r[1] == c for r in splits["test_natural"]) for c in range(C)},
        "words_per_review": {"mean": round(statistics.mean(lengths), 1),
                             "median": statistics.median(lengths),
                             "p95": sorted(lengths)[int(0.95 * len(lengths))]},
    }
    Path(stats_path(out_dir)).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
