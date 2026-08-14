"""
Clean inference wrapper around the fine-tuned DistilBERT model.
Used directly, by the FastAPI service, and by the Streamlit batch app.
"""
from pathlib import Path
from typing import List, TypedDict

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Prediction(TypedDict):
    text: str
    sentiment: str
    confidence: float


class SentimentPredictor:
    def __init__(self, model_dir: str, max_length: int = 128, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @classmethod
    def from_config(cls, config_path: str = "configs/config.yaml") -> "SentimentPredictor":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg["api"]["model_dir"], max_length=cfg["transformer"]["max_seq_length"])

    @torch.no_grad()
    def predict(self, texts: List[str], batch_size: int = 64) -> List[Prediction]:
        """Batch inference. A single string is also accepted for convenience."""
        if isinstance(texts, str):
            texts = [texts]

        results: List[Prediction] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = probs.max(dim=-1)

            for text, pred_id, conf in zip(batch, pred_ids.tolist(), confidences.tolist()):
                results.append({
                    "text": text,
                    "sentiment": self.model.config.id2label[pred_id],
                    "confidence": round(conf, 4),
                })
        return results


if __name__ == "__main__":
    predictor = SentimentPredictor.from_config()
    example = "I absolutely loved this product. The quality is amazing."
    result = predictor.predict(example)[0]
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
