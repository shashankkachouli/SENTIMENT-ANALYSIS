"""
Classical baseline: TF-IDF + Logistic Regression.

Why bother with a baseline before touching BERT? Three reasons that matter for
an ML engineering resume, not just a tutorial:
1. Sanity check: if a linear model on bag-of-words gets 85% and your fine-tuned
   transformer gets 86%, something in the pipeline (labels, leakage, splits) is
   probably wrong, or the task is largely lexical and doesn't need a transformer.
2. Cost baseline: it tells you the marginal accuracy transformers actually buy
   you, in exchange for orders of magnitude more compute -- exactly the kind of
   trade-off an ML engineer is expected to reason about.
3. Speed: a full baseline run takes seconds, so it validates the entire data
   pipeline before you spend GPU hours on the transformer.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.utils.seed import set_seed


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--compare", action="store_true",
                         help="Also train Linear SVM and Naive Bayes for comparison.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    vectorizer = TfidfVectorizer(
        max_features=cfg["baseline"]["tfidf_max_features"],
        ngram_range=tuple(cfg["baseline"]["tfidf_ngram_range"]),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train, y_test = train_df["label_name"], test_df["label_name"]

    models = {
        "logistic_regression": LogisticRegression(
            C=cfg["baseline"]["logreg_C"],
            max_iter=cfg["baseline"]["logreg_max_iter"],
            random_state=cfg["seed"],
        )
    }
    if args.compare:
        models["linear_svm"] = LinearSVC(random_state=cfg["seed"])
        models["naive_bayes"] = MultinomialNB()

    results = {}
    out_dir = Path("models/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()
        results[name] = {
            "accuracy": report["accuracy"],
            "macro_f1": f1_score(y_test, preds, average="macro"),
            "weighted_f1": f1_score(y_test, preds, average="weighted"),
            "confusion_matrix": cm,
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds))

        if name == "logistic_regression":
            joblib.dump(model, out_dir / "logreg.joblib")
            joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")

    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model + results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
