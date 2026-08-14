"""Generates 01_eda.ipynb. Run once: python notebooks/_build_eda_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# EDA: Twitter Entity Sentiment Dataset\n"
   "Exploratory analysis on `data/raw/twitter_training.csv`. Run "
   "`src/data/download.py` first if this file doesn't exist yet.")

code(
"""import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import Counter
import re

with open("../configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

COLS = cfg["data"]["columns"]
df = pd.read_csv("../data/raw/" + cfg["data"]["train_file"], names=COLS)
df.shape"""
)

md("## 1. Missing values and duplicates\n"
   "Look for: any nulls in `text`/`sentiment`, and how many exact-duplicate tweets exist "
   "(duplicates inflate class counts and can leak across train/val/test if not removed).")
code(
"""print("Missing values:\\n", df.isna().sum())
print("\\nExact duplicate tweets:", df.duplicated(subset=["text"]).sum())
print("Duplicate rate: {:.2%}".format(df.duplicated(subset=["text"]).mean()))"""
)

md("## 2. Class distribution\n"
   "Look for: how imbalanced the 4 raw labels are. This determines whether we need "
   "class weighting or stratified sampling downstream.")
code(
"""counts = df["sentiment"].value_counts()
ax = counts.plot(kind="bar", figsize=(6,4), title="Class distribution (raw labels)")
ax.set_ylabel("count")
plt.tight_layout()
plt.show()
counts / counts.sum()"""
)

md("## 3. Text length distribution (characters)\n"
   "Look for: the right tail. Twitter has a 280-char limit, so a long tail past that "
   "usually means concatenated/garbled scrape artifacts, not real tweets -- this is what "
   "the `max_chars_hard_cutoff` in preprocessing is for.")
code(
"""df["char_len"] = df["text"].astype(str).str.len()
df["char_len"].plot(kind="hist", bins=60, figsize=(6,4), title="Character length distribution")
plt.xlabel("characters")
plt.show()
df["char_len"].describe()"""
)

md("## 4. Word count distribution\n"
   "Look for: whether most tweets are very short (a handful of words), which affects "
   "the max_seq_length choice for the transformer -- if 95% of tweets are under ~40 "
   "tokens, a max_seq_length of 128 is generous, not a bottleneck.")
code(
"""df["word_count"] = df["text"].astype(str).str.split().apply(len)
df["word_count"].plot(kind="hist", bins=40, figsize=(6,4), title="Word count distribution")
plt.xlabel("words")
plt.show()
df["word_count"].quantile([0.5, 0.9, 0.95, 0.99])"""
)

md("## 5. Most common words overall, and by sentiment\n"
   "Look for: whether the top words per class are actually sentiment-bearing "
   "(e.g. 'love', 'hate') or just generic stopwords/entity names -- the latter would "
   "signal the TF-IDF baseline is picking up on entity identity rather than sentiment.")
code(
"""STOP = set("the a an is are was were to of and in on for this that i you it my your".split())

def top_words(text_series, n=20):
    words = re.findall(r"[a-zA-Z']+", " ".join(text_series).lower())
    words = [w for w in words if w not in STOP and len(w) > 2]
    return Counter(words).most_common(n)

for label in df["sentiment"].unique():
    print(f"\\n--- {label} ---")
    print(top_words(df[df["sentiment"] == label]["text"].astype(str)))"""
)

md("## 6. Outlier / very-long-text inspection\n"
   "Look at the actual longest rows manually -- confirms whether they're real "
   "long-form tweets/threads or scraping junk (repeated characters, HTML fragments, etc).")
code(
"""df.sort_values("char_len", ascending=False)[["text", "char_len", "sentiment"]].head(10)"""
)

md("## Summary of findings to carry into preprocessing\n"
   "- Fill in after running the cells above on the real downloaded data:\n"
   "  - Missing value count: ...\n"
   "  - Duplicate rate: ...\n"
   "  - Class balance (is Neutral/Irrelevant under- or over-represented?): ...\n"
   "  - Chosen `max_seq_length` based on the word-count quantiles above: ...\n"
   "  - Any garbage rows found in the long-text check that justify the hard character cutoff: ...")

nb["cells"] = cells
nbf.write(nb, "01_eda.ipynb")
print("Wrote notebooks/01_eda.ipynb")
