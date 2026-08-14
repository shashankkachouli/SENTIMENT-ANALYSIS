import pandas as pd

from src.preprocessing.clean import clean

CFG = {
    "data": {
        "label_map": {
            "Positive": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
            "Irrelevant": "Neutral",
            "max_chars_hard_cutoff": 2000,
        },
        "max_chars_hard_cutoff": 2000,
    },
    "preprocessing": {"min_text_length": 2},
}


def test_drops_missing_text():
    df = pd.DataFrame({"text": ["good tweet", None], "sentiment": ["Positive", "Negative"]})
    out = clean(df, CFG)
    assert len(out) == 1


def test_maps_irrelevant_to_neutral():
    df = pd.DataFrame({"text": ["some tweet"], "sentiment": ["Irrelevant"]})
    out = clean(df, CFG)
    assert out.iloc[0]["label_name"] == "Neutral"


def test_drops_invalid_labels():
    df = pd.DataFrame({"text": ["a tweet"], "sentiment": ["NotARealLabel"]})
    out = clean(df, CFG)
    assert len(out) == 0


def test_deduplicates_exact_text():
    df = pd.DataFrame({
        "text": ["same tweet", "same tweet", "different tweet"],
        "sentiment": ["Positive", "Positive", "Negative"],
    })
    out = clean(df, CFG)
    assert len(out) == 2


def test_drops_oversized_rows():
    df = pd.DataFrame({"text": ["x" * 5000, "normal tweet"], "sentiment": ["Positive", "Negative"]})
    out = clean(df, CFG)
    assert len(out) == 1
