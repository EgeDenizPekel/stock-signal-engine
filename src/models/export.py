"""
One-time model export script.

Pulls the best run per model type from MLflow, saves artifacts to models/,
computes test-set metrics, and writes:
  models/scaler.pkl
  models/logistic_regression.pkl
  models/random_forest.pkl
  models/xgboost.ubj
  models/lstm.pt
  models/metadata.json   — feature columns, LSTM threshold, input size
  models/metrics.json    — test-set metrics for all models

Run this once after training to prepare the API.

Usage:
    python -m src.models.export
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from .evaluate import (
    MODEL_TYPES,
    best_run_for_model,
    compute_metrics,
    find_best_threshold,
    load_sklearn_model,
    load_xgboost_model,
    load_lstm_model,
    predict_sklearn,
    predict_lstm,
)
from .train import (
    CALENDAR_COLS,
    MLFLOW_EXPERIMENT,
    PROCESSED_DIR,
    TEST_START,
    load_all,
    get_feature_cols,
    split,
    scale_features,
    build_lstm_sequences,
    SEQUENCE_LENGTH,
)
from ..backtest.strategy import sharpe_from_signals, buy_and_hold_sharpe

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    MODELS_DIR.mkdir(exist_ok=True)

    # --- Prepare data (same pipeline as evaluate.py) ---
    log.info("Loading and splitting data ...")
    df = load_all()
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split(df)
    X_train, X_val, X_test, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )
    y_test = test_df["target"].values

    continuous_cols = [c for c in feature_cols if c not in CALENDAR_COLS]
    calendar_cols   = [c for c in feature_cols if c in CALENDAR_COLS]

    # --- Save scaler ---
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    log.info("Saved scaler.pkl")

    # --- Per-ticker close prices for Sharpe ---
    raw_dir = PROCESSED_DIR.parent / "raw"
    import pandas as pd
    ticker_close: dict[str, pd.Series] = {}
    for f in sorted(raw_dir.glob("*.csv")):
        raw = pd.read_csv(f, index_col="Date", parse_dates=True)
        raw.columns = [c.lower() for c in raw.columns]
        ticker_close[f.stem] = raw.loc[TEST_START:, "close"]

    test_tickers = test_df["ticker"].values
    unique_tickers = sorted(set(test_tickers))

    # --- Export each model and compute metrics ---
    metrics_list = []
    lstm_threshold = 0.5  # default, will be updated

    for model_type in MODEL_TYPES:
        log.info("Exporting %s ...", model_type)
        try:
            run = best_run_for_model(model_type)
        except RuntimeError as e:
            log.warning("Skipping %s: %s", model_type, e)
            continue

        if model_type == "logistic_regression":
            model = load_sklearn_model(run)
            joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
            preds, proba = predict_sklearn(model, X_test)
            y_for_metrics, proba_for_metrics = y_test, proba

        elif model_type == "random_forest":
            model = load_sklearn_model(run)
            joblib.dump(model, MODELS_DIR / "random_forest.pkl")
            preds, proba = predict_sklearn(model, X_test)
            y_for_metrics, proba_for_metrics = y_test, proba

        elif model_type == "xgboost":
            model = load_xgboost_model(run)
            model.save_model(str(MODELS_DIR / "xgboost.ubj"))
            preds, proba = predict_sklearn(model, X_test)
            y_for_metrics, proba_for_metrics = y_test, proba

        elif model_type == "lstm":
            model = load_lstm_model(run, input_size=X_test.shape[1])
            torch.save(model, MODELS_DIR / "lstm.pt")

            # Find optimal threshold on val set
            _, val_proba, val_y = predict_lstm(model, X_val, val_df)
            lstm_threshold = find_best_threshold(val_proba, val_y)
            log.info("LSTM threshold: %.2f", lstm_threshold)

            _, proba, y_lstm = predict_lstm(model, X_test, test_df)
            preds = (proba >= lstm_threshold).astype(int)
            y_for_metrics, proba_for_metrics = y_lstm, proba

        else:
            continue

        metrics = compute_metrics(y_for_metrics, preds, proba_for_metrics)

        # Sharpe per ticker
        sharpe_values = []
        if model_type == "lstm":
            from .train import SEQUENCE_LENGTH as SEQ_LEN
            offset = 0
            lstm_ticker_preds: dict[str, np.ndarray] = {}
            for ticker in unique_tickers:
                n_rows = int((test_tickers == ticker).sum())
                n_seqs = n_rows - SEQ_LEN + 1
                if n_seqs > 0:
                    lstm_ticker_preds[ticker] = preds[offset: offset + n_seqs]
                    offset += n_seqs
            for ticker in unique_tickers:
                tp = lstm_ticker_preds.get(ticker)
                if tp is None or ticker not in ticker_close:
                    continue
                close = ticker_close[ticker].iloc[SEQ_LEN - 1:]
                if len(tp) == len(close):
                    sharpe_values.append(sharpe_from_signals(tp, close))
        else:
            for ticker in unique_tickers:
                if ticker not in ticker_close:
                    continue
                mask = test_tickers == ticker
                tp = preds[mask]
                close = ticker_close[ticker]
                if len(tp) == len(close):
                    sharpe_values.append(sharpe_from_signals(tp, close))

        metrics["sharpe"] = round(float(np.mean(sharpe_values)) if sharpe_values else 0.0, 3)
        metrics["model"] = model_type
        metrics_list.append(metrics)
        log.info("%s  roc_auc=%.4f  sharpe=%.3f", model_type, metrics["roc_auc"], metrics["sharpe"])

    # --- Save metrics.json ---
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_list, indent=2))
    log.info("Saved metrics.json")

    # --- Save metadata.json ---
    metadata = {
        "feature_cols": feature_cols,
        "continuous_cols": continuous_cols,
        "calendar_cols": calendar_cols,
        "lstm_threshold": lstm_threshold,
        "lstm_input_size": len(feature_cols),
        "supported_tickers": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "SPY"
        ],
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved metadata.json")

    log.info("Export complete. All artifacts in %s", MODELS_DIR)


if __name__ == "__main__":
    main()
