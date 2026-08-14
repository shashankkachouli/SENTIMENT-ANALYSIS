"""
Fine-tune DistilBERT for 3-class sentiment classification.

Why DistilBERT over full BERT (see README section 6 for the full comparison table):
DistilBERT retains ~97% of BERT's language-understanding performance on GLUE while
being ~40% smaller and ~60% faster at inference (Sanh et al., 2019, arXiv:1910.01108).
For a Twitter-length classification task -- short, mostly single-sentence text, not
long-document reasoning -- that gap in raw language understanding rarely translates
into a meaningful accuracy difference. Given this project explicitly needs to run on
free/Colab-tier GPUs and expose a low-latency inference API, the faster train/inference
time and lower memory footprint are the better engineering trade-off. If your own run
shows a large accuracy gap in BERT's favor, that's a legitimate reason to switch back --
this script's model name is a config value, not a hardcoded assumption.

Uses the Hugging Face `Trainer` for the training loop (checkpointing, early stopping,
mixed precision, and logging are all handled correctly by a well-tested library here),
but the tokenization, Dataset construction, and metrics are written explicitly so the
mechanics stay visible rather than fully hidden behind abstractions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils.seed import set_seed

LABELS = ["Negative", "Neutral", "Positive"]  # fixed, alphabetical-independent ordering
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    df = df.copy()
    df["label"] = df["label_name"].map(LABEL2ID)
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision,
        "recall_weighted": recall,
    }


def train(cfg: dict, overrides: dict | None = None) -> Trainer:
    """overrides lets Optuna (tune_optuna.py) reuse this exact function per-trial."""
    tcfg = {**cfg["transformer"], **(overrides or {})}
    set_seed(cfg["seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])
    train_ds = build_dataset(train_df, tokenizer, tcfg["max_seq_length"])
    val_ds = build_dataset(val_df, tokenizer, tcfg["max_seq_length"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg["model_name"],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["eval_batch_size"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        num_train_epochs=tcfg["num_train_epochs"],
        warmup_ratio=tcfg["warmup_ratio"],
        fp16=tcfg["fp16"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=tcfg["logging_steps"],
        report_to=["tensorboard"],
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tcfg["early_stopping_patience"])],
    )
    trainer.train()

    best_dir = Path(tcfg["output_dir"]) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = train(cfg)
    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)


if __name__ == "__main__":
    main()
