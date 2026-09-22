"""Fine-tune RoBERTa for 3-class sentiment (negative/neutral/positive).

Usage:
  python finetune_roberta.py                                    # v1: roberta-base -> models/roberta_base_3class
  python finetune_roberta.py --model roberta-large --name roberta_large_3class --lr 1e-5 --batch-size 8 --grad-accum 4 \
      --eval-steps 1875 --ckpt-dir /content/drive/MyDrive/btp_ckpt/roberta_large_3class   # v2 -> models/roberta_large_3class
  python finetune_roberta.py --name roberta_base_5class --data-dir data5     # v3: 5 classes, one per star
  python finetune_roberta.py --name roberta_base_5class_lora --data-dir data5 --lora    # v5: LoRA
  python finetune_roberta.py --name roberta_base_5class_qlora --data-dir data5 --qlora  # v6: QLoRA (4-bit base, CUDA only)
  python finetune_roberta.py --name smoke --limit 200 --epochs 1  # smoke test (keep --name smoke: never overwrite real results)

Re-running the same command resumes from the latest checkpoint in --ckpt-dir (survives Colab disconnects).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from prepare_data import evaluate_all, label_names

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="roberta-base")
ap.add_argument("--name", default="roberta_base_3class", help="key in results/metrics.json and figure names")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--max-length", type=int, default=256)
ap.add_argument("--eval-steps", type=int, default=0, help="0 = evaluate/save once per epoch")
ap.add_argument("--limit", type=int, default=0, help="cap rows per split (smoke tests)")
ap.add_argument("--out", help="default: models/<name>")
ap.add_argument("--ckpt-dir", help="default: models/<name>_checkpoints (point at Google Drive on Colab)")
ap.add_argument("--data-dir", default="data", help="data/ = 3 classes, data5/ = 5 classes")
ap.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tuning (needs `pip install peft`)")
ap.add_argument("--qlora", action="store_true", help="LoRA on a 4-bit frozen base (needs bitsandbytes, CUDA only)")
ap.add_argument("--lora-r", type=int, default=8)
ap.add_argument("--results-dir", default="results", help="where metrics/figures/logs go (per version)")
ap.add_argument("--dump-test-probs", help="save softmax probabilities for <data-dir>/test.csv to this .npy (used by detect_label_noise.py)")
args = ap.parse_args()
args.out = args.out or f"models/{args.name}"
args.ckpt_dir = args.ckpt_dir or f"models/{args.name}_checkpoints"
set_seed(42)

ds = load_dataset("csv", data_files={s: f"{args.data_dir}/{s}.csv" for s in ("train", "val")})
n_classes = max(ds["train"]["label"]) + 1
LABELS = label_names(n_classes)
print(f"{n_classes}-class task: {LABELS}")
if args.limit:
    for s in ds:
        ds[s] = ds[s].select(range(min(args.limit, len(ds[s]))))

tok = AutoTokenizer.from_pretrained(args.model)
encode = lambda b: tok(b["text"], truncation=True, max_length=args.max_length)
ds = ds.map(encode, batched=True, remove_columns=["text", "rating"]).rename_column("label", "labels")

load_kwargs = {}
if args.qlora:  # freeze the base in 4-bit; only the LoRA matrices are trained in fp16
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        # the classification head is new and must stay trainable in fp16; quantizing it to 4 bits
        # crashes bitsandbytes ("FP4 quantization state not initialized")
        llm_int8_skip_modules=["classifier"])
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=n_classes, id2label=dict(enumerate(LABELS)),
    label2id={l: i for i, l in enumerate(LABELS)}, **load_kwargs)


if args.lora or args.qlora:  # train ~1.5% of the weights; the classifier head stays trainable
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.1,
        # attention projections only: "dense" would also match the classifier head, which peft already
        # trains in full through modules_to_save
        target_modules=["query", "key", "value"]))
    model.print_trainable_parameters()


def compute_metrics(p):
    pred = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "macro_f1": f1_score(p.label_ids, pred, average="macro")}


cuda = torch.cuda.is_available()
strategy = dict(eval_strategy="steps", save_strategy="steps", eval_steps=args.eval_steps, save_steps=args.eval_steps) \
    if args.eval_steps else dict(eval_strategy="epoch", save_strategy="epoch")
targs = TrainingArguments(
    output_dir=args.ckpt_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 4,
    gradient_accumulation_steps=args.grad_accum,
    warmup_steps=0.06,  # float in [0,1) = ratio of total steps (transformers v5)
    weight_decay=0.01,
    **strategy,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    fp16=cuda,
    dataloader_num_workers=2 if cuda else 0,
    report_to="none",
    seed=42,
)
trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["val"],
                  data_collator=DataCollatorWithPadding(tok), processing_class=tok, compute_metrics=compute_metrics)

resume = any(Path(args.ckpt_dir).glob("checkpoint-*"))
print(f"Resuming from latest checkpoint in {args.ckpt_dir}" if resume else "Starting fresh")
t0 = time.time()
trainer.train(resume_from_checkpoint=resume or None)
train_seconds = round(time.time() - t0, 1)


def predict_logits(texts):
    d = Dataset.from_dict({"text": texts})
    return trainer.predict(d.map(encode, batched=True, remove_columns=["text"])).predictions


device = "cuda" if cuda else "mps" if torch.backends.mps.is_available() else "cpu"
if args.dump_test_probs:
    import pandas as pd
    from scipy.special import softmax
    texts = pd.read_csv(f"{args.data_dir}/test.csv").text.tolist()
    np.save(args.dump_test_probs, softmax(predict_logits(texts), axis=-1))
    print(f"Saved probabilities for {len(texts)} rows to {args.dump_test_probs}")

evaluate_all(args.name, predict_logits, {
    "train_seconds": train_seconds, "resumed": resume, "device": device, "train_size": len(ds["train"]),
    "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("limit", "out", "ckpt_dir")},
}, limit=args.limit, dir_=args.data_dir, results_dir=args.results_dir)

# Always save a plain fp16 RoBERTa, whatever the training method: for LoRA/QLoRA the adapter is merged
# into a freshly loaded fp16 base, so every version deploys in the same format (~249 MB).
if args.lora or args.qlora:
    from peft import PeftModel
    adapter = Path(args.out + "_adapter")
    trainer.model.save_pretrained(adapter)           # a few MB: the adapter on its own
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_classes, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)})
    final = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    print(f"Adapter alone: {sum(f.stat().st_size for f in adapter.glob('*')) / 1e6:.1f} MB")
else:
    final = trainer.model
final.half().save_pretrained(args.out)
tok.save_pretrained(args.out)

# Training curves for the report
rdir = Path(args.results_dir)
(rdir / "logs").mkdir(parents=True, exist_ok=True)
hist = trainer.state.log_history
(rdir / "logs" / f"train_log_{args.name}.json").write_text(json.dumps(hist, indent=2))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
if tr:
    plt.plot(*zip(*tr), label="train loss")
if ev:
    plt.plot(*zip(*ev), "o-", label="val loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(f"{args.model} fine-tuning loss")
plt.tight_layout(); plt.savefig(rdir / "figures" / f"loss_curve_{args.name}.png", dpi=150)
if cuda:  # peak GPU memory: the number that matters when comparing LoRA with full fine-tuning
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    mfile = Path(args.results_dir) / "metrics.json"
    metrics = json.loads(mfile.read_text())
    metrics[args.name]["peak_gpu_gb"] = peak
    metrics[args.name]["trainable_params"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    mfile.write_text(json.dumps(metrics, indent=2))
    print(f"Peak GPU memory: {peak} GB")
print(f"Saved model to {args.out}; trained in {train_seconds}s on {device}")
