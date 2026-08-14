"""
Comprehensive evaluation of the trained transformer on the held-out test set,
plus a side-by-side comparison against the TF-IDF + Logistic Regression baseline.

All numbers in the printed table come from actually running both models on
data/processed/test.csv -- nothing here is hardcoded or invented.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.training.train_transformer import ID2LABEL, LABEL2ID


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_transformer(texts, model, tokenizer, device, max_length=128, batch_size=64):
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.max(dim=-1).values.cpu().tolist())
    return [ID2LABEL[p] for p in all_preds], all_probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    processed_dir = Path(cfg["data"]["processed_dir"])
    test_df = pd.read_csv(processed_dir / "test.csv")
    y_true = test_df["label_name"].tolist()

    results = {}

    # --- Baseline ---
    baseline_dir = Path("models/baseline")
    vectorizer = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
    logreg = joblib.load(baseline_dir / "logreg.joblib")
    X_test = vectorizer.transform(test_df["text"])
    baseline_preds = logreg.predict(X_test)

    results["baseline_tfidf_logreg"] = {
        "accuracy": accuracy_score(y_true, baseline_preds),
        "macro_f1": f1_score(y_true, baseline_preds, average="macro"),
        "weighted_f1": f1_score(y_true, baseline_preds, average="weighted"),
        "classification_report": classification_report(y_true, baseline_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, baseline_preds, labels=list(LABEL2ID)).tolist(),
    }

    # --- Transformer ---
    model_dir = cfg["api"]["model_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    transformer_preds, _ = predict_transformer(
        test_df["text"].tolist(), model, tokenizer, device,
        max_length=cfg["transformer"]["max_seq_length"],
    )

    results["distilbert"] = {
        "accuracy": accuracy_score(y_true, transformer_preds),
        "macro_f1": f1_score(y_true, transformer_preds, average="macro"),
        "weighted_f1": f1_score(y_true, transformer_preds, average="weighted"),
        "classification_report": classification_report(y_true, transformer_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, transformer_preds, labels=list(LABEL2ID)).tolist(),
    }

    # --- Comparison table ---
    print(f"{'Model':<25}{'Accuracy':<12}{'Macro F1':<12}{'Weighted F1':<12}")
    for name, r in results.items():
        print(f"{name:<25}{r['accuracy']:<12.4f}{r['macro_f1']:<12.4f}{r['weighted_f1']:<12.4f}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions for error_analysis.py to reuse without recomputing.
    test_df["baseline_pred"] = baseline_preds
    test_df["transformer_pred"] = transformer_preds
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"\nSaved detailed results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
