"""
Error analysis: where does the transformer actually fail, and how does that
compare to the baseline's failures?

Requires reports/test_predictions.csv from evaluate.py (run that first).
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.evaluate import predict_transformer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--low_conf_threshold", type=float, default=0.55)
    parser.add_argument("--n_examples", type=int, default=15)
    args = parser.parse_args()
    cfg = load_config(args.config)

    reports_dir = Path("reports")
    df = pd.read_csv(reports_dir / "test_predictions.csv")

    df["baseline_correct"] = df["baseline_pred"] == df["label_name"]
    df["transformer_correct"] = df["transformer_pred"] == df["label_name"]

    # False positives / negatives per class, for the transformer
    print("=== Per-class errors (transformer) ===")
    for label in sorted(df["label_name"].unique()):
        fn = df[(df["label_name"] == label) & (df["transformer_pred"] != label)]
        fp = df[(df["label_name"] != label) & (df["transformer_pred"] == label)]
        print(f"{label}: {len(fn)} false negatives, {len(fp)} false positives")

    # Low-confidence predictions -- recompute confidence since evaluate.py didn't store it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = cfg["api"]["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    _, confidences = predict_transformer(
        df["text"].tolist(), model, tokenizer, device,
        max_length=cfg["transformer"]["max_seq_length"],
    )
    df["transformer_confidence"] = confidences

    low_conf = df.sort_values("transformer_confidence").head(args.n_examples)
    print(f"\n=== {args.n_examples} lowest-confidence predictions ===")
    print(low_conf[["text", "label_name", "transformer_pred", "transformer_confidence"]]
          .to_string(index=False))

    # Disagreement cases: reveal what each model type is good/bad at
    baseline_wins = df[df["baseline_correct"] & ~df["transformer_correct"]]
    transformer_wins = df[~df["baseline_correct"] & df["transformer_correct"]]
    print(f"\nBaseline correct, transformer wrong: {len(baseline_wins)} examples")
    print(f"Transformer correct, baseline wrong: {len(transformer_wins)} examples")

    baseline_wins.head(args.n_examples).to_csv(reports_dir / "baseline_wins.csv", index=False)
    transformer_wins.head(args.n_examples).to_csv(reports_dir / "transformer_wins.csv", index=False)
    low_conf.to_csv(reports_dir / "low_confidence_examples.csv", index=False)

    print(f"\nSaved detailed CSVs to {reports_dir.resolve()}")
    print(
        "\nManually read baseline_wins.csv / transformer_wins.csv and look for patterns: "
        "sarcasm and negation (e.g. 'not bad at all') typically favor the transformer, "
        "since TF-IDF has no notion of word order. Domain slang/hashtags absent from "
        "BERT's pretraining vocabulary, and very short/ambiguous tweets with no clear "
        "sentiment words, are common failure modes for BOTH models."
    )


if __name__ == "__main__":
    main()
