import time
from datetime import datetime, timezone

import numpy as np
import torch
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from api.main import registry
from api.schemas import SignalResponse, HistoryResponse, HistoryPoint
from src.data.features import compute_features_for_api
from src.models.lstm import SEQUENCE_LENGTH

router = APIRouter()

VALID_MODELS = ["logistic_regression", "random_forest", "xgboost", "lstm"]
_cache: dict[tuple, dict] = {}
CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cached(ticker: str, model: str) -> dict | None:
    entry = _cache.get((ticker, model))
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(ticker: str, model: str, data: dict) -> None:
    _cache[(ticker, model)] = {"data": data, "ts": time.time()}


def _scale(X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """Apply scaler to continuous columns only, pass calendar columns through."""
    meta = registry["metadata"]
    scaler = registry["scaler"]
    continuous_cols = meta["continuous_cols"]
    calendar_cols   = meta["calendar_cols"]

    cont_idx = [feature_cols.index(c) for c in continuous_cols]
    cal_idx  = [feature_cols.index(c) for c in calendar_cols]

    scaled   = scaler.transform(X[:, cont_idx])
    calendar = X[:, cal_idx]
    return np.hstack([scaled, calendar])


def _predict_rows(
    model_name: str,
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (predictions, probabilities) for every row of X_scaled.

    For LSTM, rows without a full 20-day window get probability=NaN.
    """
    model = registry["models"][model_name]

    if model_name == "lstm":
        n = len(X_scaled)
        probas = np.full(n, np.nan)
        threshold = registry["metadata"]["lstm_threshold"]
        device = next(model.parameters()).device

        for i in range(SEQUENCE_LENGTH - 1, n):
            window = X_scaled[i - SEQUENCE_LENGTH + 1: i + 1]
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probas[i] = model(x).item()

        preds = np.where(np.isnan(probas), 0, (probas >= threshold).astype(int))
        return preds, probas

    else:
        proba = model.predict_proba(X_scaled)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds, proba


def _fetch_and_compute(ticker: str) -> "pd.DataFrame":
    """Fetch ~120 days of live data and compute features."""
    raw = yf.download(ticker, period="120d", auto_adjust=True, progress=False)
    if raw.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")
    import pandas as pd
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"
    feat_df = compute_features_for_api(raw, ticker)
    if feat_df.empty:
        raise HTTPException(status_code=422, detail=f"Insufficient data to compute features for '{ticker}'")
    return feat_df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/signal/{ticker}", response_model=SignalResponse)
def get_signal(
    ticker: str,
    model: str = Query(default="logistic_regression", enum=VALID_MODELS),
):
    """Return the current buy/hold signal for a ticker."""
    ticker = ticker.upper()
    if model not in VALID_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid model '{model}'. Choose from {VALID_MODELS}")

    cached = _get_cached(ticker, model)
    if cached:
        return SignalResponse(**cached)

    feat_df = _fetch_and_compute(ticker)
    feature_cols = registry["metadata"]["feature_cols"]

    X = feat_df[feature_cols].values
    X_scaled = _scale(X, feature_cols)

    # Use only the most recent row
    _, probas = _predict_rows(model, X_scaled)
    prob = float(probas[-1]) if not np.isnan(probas[-1]) else 0.5
    threshold = registry["metadata"]["lstm_threshold"] if model == "lstm" else 0.5
    signal = "BUY" if prob >= threshold else "HOLD"

    result = {
        "ticker": ticker,
        "model": model,
        "signal": signal,
        "probability": round(prob, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _set_cached(ticker, model, result)
    return SignalResponse(**result)


@router.get("/history/{ticker}", response_model=HistoryResponse)
def get_history(
    ticker: str,
    days: int = Query(default=90, ge=20, le=365),
    model: str = Query(default="logistic_regression", enum=VALID_MODELS),
):
    """Return OHLCV + model signal for the last N trading days."""
    ticker = ticker.upper()
    if model not in VALID_MODELS:
        raise HTTPException(status_code=422, detail=f"Invalid model '{model}'. Choose from {VALID_MODELS}")

    # Fetch extra data for rolling window warmup
    fetch_days = days + 120
    raw = yf.download(ticker, period=f"{fetch_days}d", auto_adjust=True, progress=False)
    if raw.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")
    import pandas as pd
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index.name = "Date"

    feat_df = compute_features_for_api(raw, ticker)
    if len(feat_df) < days:
        raise HTTPException(status_code=422, detail=f"Not enough data for {days} days on '{ticker}'")

    feature_cols = registry["metadata"]["feature_cols"]
    X = feat_df[feature_cols].values
    X_scaled = _scale(X, feature_cols)

    preds, probas = _predict_rows(model, X_scaled)

    # Take last `days` rows
    feat_tail = feat_df.iloc[-days:]
    preds_tail = preds[-days:]
    probas_tail = probas[-days:]

    threshold = registry["metadata"]["lstm_threshold"] if model == "lstm" else 0.5

    data = []
    for i, (date, row) in enumerate(feat_tail.iterrows()):
        prob = float(probas_tail[i]) if not np.isnan(probas_tail[i]) else 0.5
        data.append(HistoryPoint(
            date=str(date.date()),
            open=round(float(row["open"]), 4),
            high=round(float(row["high"]), 4),
            low=round(float(row["low"]), 4),
            close=round(float(row["close"]), 4),
            volume=int(row["volume"]),
            signal="BUY" if prob >= threshold else "HOLD",
            probability=round(prob, 4),
        ))

    return HistoryResponse(ticker=ticker, model=model, days=days, data=data)
