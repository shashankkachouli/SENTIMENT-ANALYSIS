"""
Download the raw dataset from Kaggle.

This project cannot download Kaggle data on your behalf (it needs your personal
Kaggle API token), so this script wraps the official `kaggle` CLI instead of
silently faking data.

Setup (one-time):
    1. Create a Kaggle account -> Account -> "Create New API Token".
       This downloads kaggle.json.
    2. Place it at ~/.kaggle/kaggle.json  (chmod 600 ~/.kaggle/kaggle.json)
    3. pip install kaggle

Usage:
    python src/data/download.py --out_dir data/raw
"""
import argparse
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "jp797498e/twitter-entity-sentiment-analysis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(out_dir), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        sys.exit(
            "The `kaggle` CLI is not installed/configured. Run `pip install kaggle`, "
            "place your kaggle.json at ~/.kaggle/kaggle.json, then re-run this script."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Kaggle download failed: {e}")

    print(f"Done. Files written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
