"""Try the model on your own text.

  python predict_sentiment.py "Stopped working after a week"     # one or more reviews, prints label + confidence
  python predict_sentiment.py < reviews.txt                      # one review per line
  python predict_sentiment.py --web                              # Gradio page at http://127.0.0.1:7860

Uses models/roberta_base_5class if present, else models/roberta_base_3class. Override with MODEL=models/roberta_large_3class.
"""
import os, sys

import torch
from transformers import pipeline

from prepare_data import default_model

model = os.environ.get("MODEL") or default_model()
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
web = "--web" in sys.argv
texts = [a for a in sys.argv[1:] if a != "--web"]
# top_k=None returns every class (the web page shows all); the CLI wants just the best label
clf = pipeline("text-classification", model=model, device=device, **({"top_k": None} if web else {}))
print(f"model: {model}", file=sys.stderr)

if web:
    import gradio as gr

    gr.Interface(
        fn=lambda text: {o["label"]: o["score"] for o in clf([text], truncation=True, max_length=256)[0]},
        inputs=gr.Textbox(lines=4, label="Product review"),
        outputs=gr.Label(num_top_classes=5, label="Sentiment"),
        title="Amazon Industrial & Scientific — Review Sentiment",
        description=f"Fine-tuned `{model}` (RoBERTa).",
        examples=[["Broke after two days, total junk. Returning it."],
                  ["It's okay, does the job but feels cheap."],
                  ["Works exactly as described, great value for the price."],
                  ["The pH meter reads 0.3 off even after calibration."]],
        flagging_mode="never",
    ).launch()
else:
    def show(batch):
        for text, out in zip(batch, clf(batch, truncation=True, max_length=256)):
            print(f"{out['label']:>18}  {out['score']:.3f}  {text}")

    if texts:
        show(texts)
    else:
        print("Type a review and press Enter (Ctrl-D to quit).", file=sys.stderr)
        for line in sys.stdin:
            if line.strip():
                show([line.strip()])
