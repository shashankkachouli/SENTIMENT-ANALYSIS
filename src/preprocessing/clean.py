"""
Production-quality cleaning + splitting for the Twitter entity-sentiment dataset.

Design notes (see README section 4 for the full explanation):
- We do NOT lowercase, strip punctuation, remove stopwords, or stem/lemmatize.
  BERT-family tokenizers (WordPiece) are trained on natural text including casing
  and punctuation; stripping it throws away signal the model was pretrained on
  and can hurt performance. Classical preprocessing rules are for TF-IDF, not BERT.
- What we DO fix: missing text, exact duplicates, invalid/unmapped labels,
  broken encoding, and pathologically long rows that are almost always scraping
  artifacts rather than real tweets.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(raw_dir: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    # The Kaggle files ship with NO header row.
    df = pd.read_csv(raw_dir / filename, names=columns, encoding="utf-8", encoding_errors="replace")
    return df


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n_start = len(df)

    # 1. Missing values
    df = df.dropna(subset=["text", "sentiment"]).copy()

    # 2. Normalize whitespace only (safe for any downstream model)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= cfg["preprocessing"]["min_text_length"]]

    # 3. Invalid / unmapped labels
    label_map = cfg["data"]["label_map"]
    df = df[df["sentiment"].isin(label_map.keys())].copy()
    df["label_name"] = df["sentiment"].map(label_map)

    # 4. Duplicates (exact text duplicates, keep first)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    n_dupes = n_before_dedup - len(df)

    # 5. Pathologically long rows (parsing artifacts, not real tweets)
    cutoff = cfg["data"]["max_chars_hard_cutoff"]
    n_before_cutoff = len(df)
    df = df[df["text"].str.len() <= cutoff]
    n_too_long = n_before_cutoff - len(df)

    print(f"[clean] start={n_start} -> end={len(df)} "
          f"(dropped {n_dupes} dupes, {n_too_long} oversized rows)")
    return df.reset_index(drop=True)


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Data leakage here = the exact same tweet text appearing in more than one split."""
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    overlaps = {
        "train/val": train_texts & val_texts,
        "train/test": train_texts & test_texts,
        "val/test": val_texts & test_texts,
    }
    for name, overlap in overlaps.items():
        if overlap:
            print(f"[leakage-check] WARNING: {len(overlap)} overlapping texts between {name}")
        else:
            print(f"[leakage-check] OK: no overlap between {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = cfg["data"]["columns"]

    train_raw = load_raw(raw_dir, cfg["data"]["train_file"], columns)
    val_raw = load_raw(raw_dir, cfg["data"]["val_file"], columns)

    train_df = clean(train_raw, cfg)
    val_full_df = clean(val_raw, cfg)

    # The Kaggle release only ships train + validation. We split the provided
    # validation file in half (stratified) to get an untouched held-out test set,
    # so hyperparameter tuning never sees the final test data.
    from sklearn.model_selection import train_test_split

    val_df, test_df = train_test_split(
        val_full_df,
        test_size=cfg["preprocessing"]["test_size_from_val"],
        stratify=val_full_df["label_name"],
        random_state=cfg["seed"],
    )

    check_leakage(train_df, val_df, test_df)

    print("\n[class balance] train:")
    print(train_df["label_name"].value_counts(normalize=True).round(3))

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    print(f"\nWrote train/val/test to {processed_dir.resolve()}")
    print(f"Sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    main()
