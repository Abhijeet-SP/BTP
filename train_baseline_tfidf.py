"""TF-IDF + Logistic Regression baseline on the same splits as RoBERTa.

Usage: python train_baseline_tfidf.py [--data-dir data5]
"""
import argparse, time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from prepare_data import evaluate_all

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data", help="data/ = 3 classes, data5/ = 5 classes")
ap.add_argument("--name", default=None, help="metrics key; default tfidf_3class / tfidf_5class")
ap.add_argument("--results-dir", default="results")
args = ap.parse_args()
name = args.name or ("tfidf_3class" if args.data_dir == "data" else "tfidf_5class")

train = pd.read_csv(f"{args.data_dir}/train.csv")

t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(vec.fit_transform(train.text), train.label)

evaluate_all(name, lambda texts: clf.predict_log_proba(vec.transform(texts)),
             {"train_seconds": round(time.time() - t0, 1)}, dir_=args.data_dir, results_dir=args.results_dir)
