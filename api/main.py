"""
FastAPI service exposing the sentiment model.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"texts": ["I love this!", "This is terrible."]}'
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference.predict import SentimentPredictor

MAX_BATCH_SIZE = 256
MAX_TEXT_LENGTH = 2000

predictor_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per-request.
    try:
        predictor_holder["predictor"] = SentimentPredictor.from_config()
    except Exception as e:  # noqa: BLE001 - surfaced via /health, not swallowed
        predictor_holder["load_error"] = str(e)
    yield
    predictor_holder.clear()


app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        cleaned = [t.strip() for t in texts]
        if any(len(t) == 0 for t in cleaned):
            raise ValueError("texts must not contain empty strings")
        if any(len(t) > MAX_TEXT_LENGTH for t in cleaned):
            raise ValueError(f"each text must be <= {MAX_TEXT_LENGTH} characters")
        return cleaned


class PredictionItem(BaseModel):
    text: str
    sentiment: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]


@app.get("/health")
def health():
    if "predictor" not in predictor_holder:
        return {"status": "error", "detail": predictor_holder.get("load_error", "model not loaded")}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictor: SentimentPredictor | None = predictor_holder.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")
    try:
        results = predictor.predict(request.texts)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    return {"predictions": results}
