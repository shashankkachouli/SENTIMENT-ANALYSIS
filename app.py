
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Configuration
# -----------------------------
## MODEL_PATH = "/content/drive/MyDrive/sentiment_project/models/distilbert-sentiment/best"
MODEL_PATH = "kalanag/twitter-sentiment-distilbert"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()

# -----------------------------
# Prediction
# -----------------------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_id = torch.argmax(probabilities).item()

    return (
        LABELS[predicted_id],
        probabilities.cpu().numpy()
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Twitter Sentiment Analyzer")

st.write(
    "Fine-tuned DistilBERT sentiment classification model"
)

st.info(
    "Model: DistilBERT | "
    "Classes: Negative, Neutral, Positive"
)

text = st.text_area(
    "Enter a tweet or text:",
    placeholder="Example: I absolutely love this product!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        prediction, probabilities = predict_sentiment(text)

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success(f"😊 {prediction}")

        elif prediction == "Negative":
            st.error(f"😠 {prediction}")

        else:
            st.info(f"😐 {prediction}")

        st.subheader("Confidence")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Negative",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Neutral",
                f"{probabilities[1] * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Positive",
                f"{probabilities[2] * 100:.2f}%"
            )

        st.progress(
            float(probabilities.max()),
            text=f"Model confidence: {probabilities.max() * 100:.2f}%"
        )

st.divider()

st.caption(
    "Fine-tuned DistilBERT • 97.2% test accuracy • 97.2% Macro F1"
)
