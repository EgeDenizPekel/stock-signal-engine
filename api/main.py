"""
FastAPI application entry point.

Loads all model artifacts at startup from models/ directory.
All models are held in `registry` and shared across routers.

Usage:
    uvicorn api.main:app --reload
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

registry: dict = {
    "models": {},
    "scaler": None,
    "metadata": {},
    "metrics": [],
}


def _load_registry() -> None:
    """Load all model artifacts from models/ into memory."""
    if not MODELS_DIR.exists():
        raise RuntimeError(
            "models/ directory not found. Run `python -m src.models.export` first."
        )

    registry["scaler"]   = joblib.load(MODELS_DIR / "scaler.pkl")
    registry["metadata"] = json.loads((MODELS_DIR / "metadata.json").read_text())
    registry["metrics"]  = json.loads((MODELS_DIR / "metrics.json").read_text())

    registry["models"]["logistic_regression"] = joblib.load(
        MODELS_DIR / "logistic_regression.pkl"
    )
    registry["models"]["random_forest"] = joblib.load(
        MODELS_DIR / "random_forest.pkl"
    )

    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost.ubj"))
    registry["models"]["xgboost"] = xgb_model

    from src.models.lstm import LSTMClassifier
    lstm = torch.load(MODELS_DIR / "lstm.pt", map_location="cpu", weights_only=False)
    lstm.eval()
    registry["models"]["lstm"] = lstm

    log.info("Registry loaded: %d models", len(registry["models"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_registry()
    yield


app = FastAPI(
    title="Stock Signal Engine",
    description="ML-powered buy/hold signals for major US stocks.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from api.routers import signal, performance  # noqa: E402

app.include_router(signal.router)
app.include_router(performance.router)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(registry["models"].keys())}
