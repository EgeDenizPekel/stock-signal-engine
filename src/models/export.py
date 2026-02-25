"""
One-time model export script.

Pulls the best run per model type from MLflow, saves artifacts to models/,
computes test-set and val-set metrics, and writes:
  models/scaler.pkl
  models/logistic_regression.pkl
  models/random_forest.pkl
  models/xgboost.ubj
  models/lstm.pt
  models/metadata.json   — feature columns, LSTM threshold, input size
  models/metrics.json    — {"test": [...], "val": [...]} metrics for all models

Run this once after training to prepare the API.

Usage:
    python -m src.models.export
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    VAL_START,
    VAL_END,
    TEST_START,
    load_all,
    get_feature_cols,
    split,
    scale_features,
    build_lstm_sequences,
    SEQUENCE_LENGTH,
)
from ..backtest.strategy import (
    daily_strategy_returns, sharpe_ratio, buy_and_hold_sharpe,
    cagr, max_drawdown, sortino_ratio, win_rate, turnover,
    avg_gain, avg_loss, payoff_ratio,
)

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _backtest_metrics(
    model_type: str,
    preds: np.ndarray,
    tickers: np.ndarray,
    ticker_close_dict: dict,
    unique_tickers: list,
    seq_len: int = 0,
) -> dict:
    """
    Run per-ticker backtest and return averaged trading metrics.

    For LSTM (seq_len > 0), applies the sequence-length offset when mapping
    predictions back to per-ticker close price windows.
    """
    sharpe_vals, sortino_vals, cagr_vals, dd_vals, wr_vals, to_vals = [], [], [], [], [], []
    ag_vals, al_vals, pr_vals = [], [], []

    # Build per-ticker prediction arrays (LSTM needs sequence offset)
    lstm_ticker_preds: dict[str, np.ndarray] = {}
    if model_type == "lstm" and seq_len > 0:
        offset = 0
        for ticker in unique_tickers:
            n_rows = int((tickers == ticker).sum())
            n_seqs = n_rows - seq_len + 1
            if n_seqs > 0:
                lstm_ticker_preds[ticker] = preds[offset: offset + n_seqs]
                offset += n_seqs

    for ticker in unique_tickers:
        if ticker not in ticker_close_dict:
            continue

        if model_type == "lstm" and seq_len > 0:
            tp = lstm_ticker_preds.get(ticker)
            if tp is None:
                continue
            close = ticker_close_dict[ticker].iloc[seq_len - 1:]
        else:
            mask = tickers == ticker
            tp = preds[mask]
            close = ticker_close_dict[ticker]

        if len(tp) != len(close):
            continue

        rets = daily_strategy_returns(tp, close)
        sharpe_vals.append(sharpe_ratio(rets))
        sortino_vals.append(sortino_ratio(rets))
        cagr_vals.append(cagr(rets))
        dd_vals.append(max_drawdown(rets))
        wr_vals.append(win_rate(rets))
        to_vals.append(turnover(tp))
        ag_vals.append(avg_gain(rets))
        al_vals.append(avg_loss(rets))
        pr_vals.append(payoff_ratio(rets))

    def _mean(vals):
        return round(float(np.nanmean(vals)), 5) if vals else 0.0

    return {
        "sharpe":       round(float(np.nanmean(sharpe_vals))  if sharpe_vals  else 0.0, 3),
        "sortino":      round(float(np.nanmean(sortino_vals)) if sortino_vals else 0.0, 3),
        "cagr":         round(float(np.nanmean(cagr_vals))    if cagr_vals    else 0.0, 4),
        "max_drawdown": round(float(np.nanmean(dd_vals))      if dd_vals      else 0.0, 4),
        "win_rate":     round(float(np.nanmean(wr_vals))      if wr_vals      else 0.0, 4),
        "turnover":     round(float(np.nanmean(to_vals))      if to_vals      else 0.0, 4),
        "avg_gain":     _mean(ag_vals),
        "avg_loss":     _mean(al_vals),
        "payoff_ratio": round(float(np.nanmean(pr_vals))      if pr_vals      else 0.0, 3),
    }


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    MODELS_DIR.mkdir(exist_ok=True)

    # --- Prepare data ---
    log.info("Loading and splitting data ...")
    df = load_all()
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split(df)
    X_train, X_val, X_test, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )
    y_test = test_df["target"].values
    y_val  = val_df["target"].values

    continuous_cols = [c for c in feature_cols if c not in CALENDAR_COLS]
    calendar_cols   = [c for c in feature_cols if c in CALENDAR_COLS]

    # --- Save scaler ---
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    log.info("Saved scaler.pkl")

    # --- Per-ticker close prices (test = 2024, val = 2023) ---
    raw_dir = PROCESSED_DIR.parent / "raw"
    ticker_test_close: dict[str, pd.Series] = {}
    ticker_val_close: dict[str, pd.Series] = {}
    for f in sorted(raw_dir.glob("*.csv")):
        raw = pd.read_csv(f, index_col="Date", parse_dates=True)
        raw.columns = [c.lower() for c in raw.columns]
        ticker_test_close[f.stem] = raw.loc[TEST_START:, "close"]
        ticker_val_close[f.stem]  = raw.loc[VAL_START:VAL_END, "close"]

    test_tickers   = test_df["ticker"].values
    val_tickers    = val_df["ticker"].values
    unique_tickers = sorted(set(test_tickers))
    unique_val_tickers = sorted(set(val_tickers))

    # --- Export each model and compute metrics ---
    test_metrics_list = []
    val_metrics_list  = []
    lstm_threshold = 0.5

    for model_type in MODEL_TYPES:
        log.info("Exporting %s ...", model_type)
        try:
            run = best_run_for_model(model_type)
        except RuntimeError as e:
            log.warning("Skipping %s: %s", model_type, e)
            continue

        # ----------------------------------------------------------------
        # Load model + get predictions for test and val
        # ----------------------------------------------------------------
        if model_type == "logistic_regression":
            model = load_sklearn_model(run)
            joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
            test_preds, test_proba = predict_sklearn(model, X_test)
            val_preds,  val_proba  = predict_sklearn(model, X_val)
            y_test_cls, y_val_cls  = y_test, y_val

        elif model_type == "random_forest":
            model = load_sklearn_model(run)
            joblib.dump(model, MODELS_DIR / "random_forest.pkl")
            test_preds, test_proba = predict_sklearn(model, X_test)
            val_preds,  val_proba  = predict_sklearn(model, X_val)
            y_test_cls, y_val_cls  = y_test, y_val

        elif model_type == "xgboost":
            model = load_xgboost_model(run)
            model.save_model(str(MODELS_DIR / "xgboost.ubj"))
            test_preds, test_proba = predict_sklearn(model, X_test)
            val_preds,  val_proba  = predict_sklearn(model, X_val)
            y_test_cls, y_val_cls  = y_test, y_val

        elif model_type == "lstm":
            model = load_lstm_model(run, input_size=X_test.shape[1])
            torch.save(model, MODELS_DIR / "lstm.pt")

            # Find optimal threshold on val, then apply to both sets
            _, raw_val_proba, val_y_seq = predict_lstm(model, X_val, val_df)
            lstm_threshold = find_best_threshold(raw_val_proba, val_y_seq)
            log.info("LSTM threshold: %.2f", lstm_threshold)

            _, raw_test_proba, test_y_seq = predict_lstm(model, X_test, test_df)
            test_preds = (raw_test_proba >= lstm_threshold).astype(int)
            test_proba = raw_test_proba
            y_test_cls = test_y_seq

            val_preds = (raw_val_proba >= lstm_threshold).astype(int)
            val_proba = raw_val_proba
            y_val_cls = val_y_seq

        else:
            continue

        # ----------------------------------------------------------------
        # Classification metrics
        # ----------------------------------------------------------------
        test_cls = compute_metrics(y_test_cls, test_preds, test_proba)
        val_cls  = compute_metrics(y_val_cls,  val_preds,  val_proba)

        # ----------------------------------------------------------------
        # Trading / backtest metrics
        # ----------------------------------------------------------------
        seq_len = SEQUENCE_LENGTH if model_type == "lstm" else 0

        test_bt = _backtest_metrics(
            model_type, test_preds, test_tickers,
            ticker_test_close, unique_tickers, seq_len,
        )
        val_bt = _backtest_metrics(
            model_type, val_preds, val_tickers,
            ticker_val_close, unique_val_tickers, seq_len,
        )

        # ----------------------------------------------------------------
        # Merge and collect
        # ----------------------------------------------------------------
        test_entry = {**test_cls, **test_bt, "model": model_type}
        val_entry  = {**val_cls,  **val_bt,  "model": model_type}

        test_metrics_list.append(test_entry)
        val_metrics_list.append(val_entry)

        log.info(
            "%s  test roc_auc=%.4f sharpe=%.3f cagr=%.2f%%  |"
            "  val roc_auc=%.4f sharpe=%.3f cagr=%.2f%%",
            model_type,
            test_entry["roc_auc"], test_entry["sharpe"], test_entry["cagr"] * 100,
            val_entry["roc_auc"],  val_entry["sharpe"],  val_entry["cagr"]  * 100,
        )

    # --- Save metrics.json ---
    metrics_out = {"test": test_metrics_list, "val": val_metrics_list}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
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
