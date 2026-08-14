"""
Batch Sentiment Analysis demo.

Workflow: upload .txt (one tweet per line) -> validate -> batch tokenize ->
transformer inference -> aggregate -> show Positive/Negative distribution + chart.

Shows ONLY the aggregate result, per spec: no per-tweet rows, no CSV download,
no per-prediction confidence table.

Run:
    streamlit run app/app.py
"""
import matplotlib.pyplot as plt
import streamlit as st

from src.inference.predict import SentimentPredictor

st.set_page_config(page_title="Batch Sentiment Analysis", layout="centered")


@st.cache_resource
def load_predictor() -> SentimentPredictor:
    # Loads the already fine-tuned model + tokenizer. Never retrains here.
    return SentimentPredictor.from_config()


def parse_and_validate(raw_bytes: bytes) -> tuple[list[str], list[str]]:
    """Returns (valid_tweets, warnings)."""
    warnings = []
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
            warnings.append("File was not valid UTF-8; decoded as Latin-1.")
        except UnicodeDecodeError:
            return [], ["Could not decode file. Please upload a plain UTF-8 .txt file."]

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines

    if not lines:
        return [], ["The uploaded file is empty (no non-blank lines found)."]

    MAX_TWEET_LEN = 2000
    too_long = [l for l in lines if len(l) > MAX_TWEET_LEN]
    lines = [l for l in lines if len(l) <= MAX_TWEET_LEN]
    if too_long:
        warnings.append(f"Skipped {len(too_long)} line(s) longer than {MAX_TWEET_LEN} characters.")

    n_before_dedup = len(lines)
    lines = list(dict.fromkeys(lines))  # de-duplicate, preserve order
    if n_before_dedup - len(lines) > 0:
        warnings.append(f"Removed {n_before_dedup - len(lines)} duplicate line(s).")

    return lines, warnings


def main():
    st.title("Batch Sentiment Analysis")
    st.write("Upload Tweets (.txt)")

    uploaded_file = st.file_uploader("", type=["txt"])
    if uploaded_file is None:
        return

    tweets, warnings = parse_and_validate(uploaded_file.read())
    for w in warnings:
        st.warning(w)

    if not tweets:
        st.error("No valid tweets found in the file. Please check the file and try again.")
        return

    with st.spinner(f"Running sentiment model on {len(tweets)} tweets..."):
        try:
            predictor = load_predictor()
            predictions = predictor.predict(tweets)  # batch inference, one call
        except Exception as e:  # noqa: BLE001
            st.error(f"Model inference failed: {e}")
            return

    counts: dict[str, int] = {}
    for p in predictions:
        counts[p["sentiment"]] = counts.get(p["sentiment"], 0) + 1

    # Per project spec: only show classes that are actually present in the
    # trained label set for this data (no artificial "Neutral" padding, and if
    # Neutral genuinely has zero predictions here we still only show what the
    # model actually produced).
    total = len(predictions)
    st.header("Sentiment Analysis Results")
    st.metric("Total Tweets", total)

    cols = st.columns(len(counts))
    for col, (label, count) in zip(cols, sorted(counts.items())):
        pct = 100 * count / total
        col.metric(label, f"{pct:.1f}%", f"{count} tweets")

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]
    ax.bar(labels, values)
    ax.set_ylabel("Number of tweets")
    ax.set_title("Sentiment Distribution")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    st.pyplot(fig)


if __name__ == "__main__":
    main()
