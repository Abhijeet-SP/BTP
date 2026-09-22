"""Score a trained model locally (inference only, light on a laptop).

  python evaluate_model.py --model models/roberta_base_5class --data-dir data5 --results-dir versions/v3_star_level_5class/results
  python evaluate_model.py --model models/roberta_base_5class --quantize     # int8 size/speed/accuracy trade-off (CPU)

Writes metrics under "<model>_local*", the most confident mistakes to errors/, and with --quantize
a "<model>_quantized" entry comparing fp32 against int8.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare_data import evaluate_all, label_names

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/roberta_base_3class")
ap.add_argument("--data-dir", default="data", help="data/ = 3 classes, data5/ = 5 classes")
ap.add_argument("--results-dir", default="results")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--quantize", action="store_true", help="also measure int8 dynamic quantization (CPU only)")
args = ap.parse_args()
name = Path(args.model).name
tok = AutoTokenizer.from_pretrained(args.model)


def batched_logits(model, texts, device):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(device)
            out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out)


if args.quantize:
    # Dynamic int8 for Linear AND Embedding: the 50k x 768 word table is ~40% of the weights, so
    # quantizing only the Linear layers barely shrinks the file.
    from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines \
        else torch.backends.quantized.supported_engines[0]
    test = pd.read_csv(f"{args.data_dir}/test.csv", nrows=args.limit or 4000)
    texts, labels = test.text.tolist(), test.label.to_numpy()

    def score(model):
        t0 = time.time()
        p = batched_logits(model, texts, "cpu").argmax(-1)
        return {"accuracy": round(accuracy_score(labels, p), 4),
                "macro_f1": round(f1_score(labels, p, average="macro"), 4),
                "qwk": round(cohen_kappa_score(labels, p, weights="quadratic"), 4),
                "mae_classes": round(float(np.abs(labels - p).mean()), 4),
                "reviews_per_second": round(len(texts) / (time.time() - t0), 1)}

    fp32 = AutoModelForSequenceClassification.from_pretrained(args.model).float().eval()
    out = {"fp32_cpu": score(fp32),
           "fp32_size_mb": round(sum(f.stat().st_size for f in Path(args.model).glob("*.safetensors")) / 1e6)}
    int8 = torch.ao.quantization.quantize_dynamic(
        fp32, {torch.nn.Linear: default_dynamic_qconfig, torch.nn.Embedding: float_qparams_weight_only_qconfig},
        dtype=torch.qint8)
    out["int8_cpu"] = score(int8)
    dst = Path(f"{args.model}_int8.pt")
    torch.save(int8.state_dict(), dst)
    out |= {"int8_size_mb": round(dst.stat().st_size / 1e6), "model": args.model, "test_rows": len(texts)}

    mfile = Path(args.results_dir) / "metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
    metrics[f"{name}_quantized"] = out
    mfile.parent.mkdir(parents=True, exist_ok=True)
    mfile.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(out, indent=2))
    raise SystemExit

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained(args.model)
model = (model.float() if device == "cpu" else model.half()).to(device).eval()  # fp16 matmuls are slow on CPU
n_seen, secs = 0, 0.0


def predict_logits(texts):
    global n_seen, secs
    t0 = time.time()
    logits = batched_logits(model, texts, device)
    n_seen, secs = n_seen + len(texts), secs + time.time() - t0
    return logits


size_mb = round(sum(f.stat().st_size for f in Path(args.model).glob("*.safetensors")) / 1e6)
test, logits = evaluate_all(f"{name}_local", predict_logits, {"device": device, "model_size_mb": size_mb},
                            limit=args.limit, dir_=args.data_dir, results_dir=args.results_dir)
print(f"{n_seen} reviews in {secs:.0f}s = {n_seen / secs:.1f} reviews/s on {device}; model {size_mb} MB")

# Most confident mistakes: often label noise (e.g. a 1-star review whose text is clearly positive)
probs = torch.tensor(logits).softmax(-1).numpy()
names = label_names(model.config.num_labels)
test["pred"], test["confidence"] = [names[p] for p in probs.argmax(-1)], probs.max(-1)
test["label"] = [names[l] for l in test.label]
wrong = test[test.label != test.pred].sort_values("confidence", ascending=False)
errors = Path(args.results_dir) / "errors"
errors.mkdir(parents=True, exist_ok=True)
wrong[["rating", "label", "pred", "confidence", "text"]].head(50).to_csv(errors / f"errors_{name}.csv", index=False)
