"""Find and drop label noise in the training set, then write a cleaned copy.

Star ratings are noisy labels: 1-star reviews that read as glowing, 5-star reviews titled "One Star".
A model trained on contradictions learns worse, so we remove the clearest ones.

Method (confident learning, 2-fold cross-fitting):
  1. Split train.csv into two halves.
  2. Train on half A, predict half B; train on half B, predict half A.
     Cross-fitting matters: a model that trained on a review has partly memorised it, so its own
     prediction there cannot tell noise from memorisation.
  3. Drop a review when the out-of-sample model is confident (p > --threshold) AND its prediction is
     at least --min-distance classes away from the label. For 5 ordered classes, 4-vs-5 stars is
     ambiguity rather than noise, so the default distance of 2 keeps those and removes only contradictions.
  4. Write <data-dir>_clean/ with the cleaned train.csv; val/test/test_natural are copied UNCHANGED,
     so the cleaned model is still judged on the original, untouched benchmark.

Usage: python detect_label_noise.py --data-dir data5 [--fold-epochs 1] [--threshold 0.9] [--min-distance 2]
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import label_names, stats_path

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data5")
ap.add_argument("--model", default="roberta-base")
ap.add_argument("--fold-epochs", type=float, default=1, help="1 is enough to spot contradictions")
ap.add_argument("--threshold", type=float, default=0.9)
ap.add_argument("--min-distance", type=int, default=2, help="classes apart before a disagreement counts as noise")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--ckpt-dir", default="models", help="parent for fold checkpoints (use Drive on Colab)")
ap.add_argument("--results-dir", default="results")
args = ap.parse_args()

src = Path(args.data_dir)
train = pd.read_csv(src / "train.csv")
half = len(train) // 2
folds = {"a": train.iloc[:half].reset_index(drop=True), "b": train.iloc[half:].reset_index(drop=True)}

probs = {}
for name, other in (("a", "b"), ("b", "a")):
    d = Path(f"{args.data_dir}_fold_{name}")
    d.mkdir(exist_ok=True)
    folds[name].to_csv(d / "train.csv", index=False)          # train on this half
    folds[other].to_csv(d / "test.csv", index=False)          # predict the other half
    shutil.copy(src / "val.csv", d / "val.csv")
    out = d / "probs.npy"
    if not out.exists():
        cmd = [sys.executable, "finetune_roberta.py", "--model", args.model, "--name", f"fold_{name}",
               "--data-dir", str(d), "--epochs", str(args.fold_epochs), "--batch-size", str(args.batch_size),
               "--out", f"models/fold_{name}", "--ckpt-dir", f"{args.ckpt_dir}/fold_{name}_checkpoints",
               "--dump-test-probs", str(out), "--results-dir", args.results_dir]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    probs[other] = np.load(out)                                # out-of-sample probabilities for the other half

p = np.concatenate([probs["a"], probs["b"]])                   # aligned with train row order
pred, conf = p.argmax(-1), p.max(-1)
label = train.label.to_numpy()
noisy = (pred != label) & (conf > args.threshold) & (np.abs(pred - label) >= args.min_distance)

dst = Path(f"{args.data_dir}_clean")
dst.mkdir(exist_ok=True)
train[~noisy].to_csv(dst / "train.csv", index=False)
for f in ("val.csv", "test.csv", "test_natural.csv"):         # benchmark stays untouched
    if (src / f).exists():
        shutil.copy(src / f, dst / f)
shutil.copy(stats_path(str(src)), stats_path(str(dst)))        # class priors are unchanged

names = label_names(int(max(label.max(), pred.max())) + 1)
flagged = train[noisy].copy()
flagged["predicted"] = [names[i] for i in pred[noisy]]
flagged["confidence"] = conf[noisy]
flagged["label_name"] = [names[i] for i in label[noisy]]
rdir = Path(args.results_dir)
(rdir / "errors").mkdir(parents=True, exist_ok=True)
flagged[["rating", "label_name", "predicted", "confidence", "text"]].head(100).to_csv(
    rdir / "errors" / "label_noise_sample.csv", index=False)

summary = {
    "train_rows": len(train),
    "removed": int(noisy.sum()),
    "removed_pct": round(100 * noisy.mean(), 2),
    "threshold": args.threshold,
    "min_distance": args.min_distance,
    "removed_by_star": flagged.rating.value_counts().sort_index().to_dict(),
    "removed_by_class": flagged.label_name.value_counts().to_dict(),
}
(rdir / "label_noise.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"Cleaned training set: {dst}/train.csv ({len(train) - int(noisy.sum())} rows)")
