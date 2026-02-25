"""
Model evaluation script.

Loads the best run per model type from MLflow, generates predictions on the
held-out test set (2024), computes classification metrics and backtested
Sharpe ratios, and prints a comparison table.

Usage:
    python -m src.models.evaluate
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ..backtest.strategy import sharpe_from_signals, buy_and_hold_sharpe
from .train import (
    MLFLOW_EXPERIMENT,
    CALENDAR_COLS,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    PROCESSED_DIR,
    load_all,
    get_feature_cols,
    split,
    scale_features,
)
from .lstm import LSTMClassifier, SequenceDataset, SEQUENCE_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_TYPES = ["logistic_regression", "random_forest", "xgboost", "lstm"]


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def best_run_for_model(model_type: str) -> mlflow.entities.Run:
    """
    Return the MLflow run with the highest val_roc_auc for a given model type.
    For LSTM, uses val_roc_auc_best.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_type = '{model_type}'",
        order_by=["metrics.val_roc_auc DESC"],
    )
    if not runs:
        raise RuntimeError(f"No runs found for model_type='{model_type}'.")
    return runs[0]


def load_sklearn_model(run: mlflow.entities.Run):
    return mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")


def load_xgboost_model(run: mlflow.entities.Run):
    return mlflow.xgboost.load_model(f"runs:/{run.info.run_id}/model")


def load_lstm_model(run: mlflow.entities.Run, input_size: int) -> LSTMClassifier:
    model = mlflow.pytorch.load_model(f"runs:/{run.info.run_id}/model")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict_sklearn(model, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def predict_lstm(
    model: LSTMClassifier,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-ticker sequences (matching training), run inference, and return
    predictions, probabilities, and the corresponding trimmed y_true labels.
    """
    from .train import build_lstm_sequences
    X_seq, y_seq = build_lstm_sequences(test_df, X_test, test_df["target"].values)

    device = next(model.parameters()).device
    X_tensor = torch.tensor(X_seq)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor),
        batch_size=256, shuffle=False,
    )

    all_proba = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            proba = model(X_batch)
            all_proba.extend(proba.cpu().numpy())

    proba_arr = np.array(all_proba)
    preds_arr = (proba_arr >= 0.5).astype(int)
    return preds_arr, proba_arr, y_seq


def find_best_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    Sweep thresholds from 0.2 to 0.8 and return the one that maximises F1.
    Used to calibrate the LSTM decision boundary on the validation set.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, preds), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_true, preds, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, proba), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # --- Load and prepare test data ---
    log.info("Loading data ...")
    df = load_all()
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split(df)
    X_train, X_val, X_test, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )
    y_test = test_df["target"].values

    # Build per-ticker close price series from raw CSVs for backtesting.
    # Keyed by ticker symbol → pd.Series of close prices for the test period.
    raw_dir = PROCESSED_DIR.parent / "raw"
    ticker_close: dict[str, pd.Series] = {}
    for f in sorted(raw_dir.glob("*.csv")):
        raw = pd.read_csv(f, index_col="Date", parse_dates=True)
        raw.columns = [c.lower() for c in raw.columns]
        ticker_close[f.stem] = raw.loc[TEST_START:, "close"]

    # Keep the test DataFrame index (ticker + date) to map predictions back
    # to individual tickers for per-ticker Sharpe calculation.
    test_tickers = test_df["ticker"].values

    results = []

    for model_type in MODEL_TYPES:
        log.info("Evaluating %s ...", model_type)
        try:
            run = best_run_for_model(model_type)
        except RuntimeError as e:
            log.warning("%s", e)
            continue

        # Load model and predict
        if model_type == "logistic_regression":
            model = load_sklearn_model(run)
            preds, proba = predict_sklearn(model, X_test)
            y_test_for_metrics = y_test
            proba_for_metrics  = proba

        elif model_type == "random_forest":
            model = load_sklearn_model(run)
            preds, proba = predict_sklearn(model, X_test)
            y_test_for_metrics = y_test
            proba_for_metrics  = proba

        elif model_type == "xgboost":
            model = load_xgboost_model(run)
            preds, proba = predict_sklearn(model, X_test)
            y_test_for_metrics = y_test
            proba_for_metrics  = proba

        elif model_type == "lstm":
            model = load_lstm_model(run, input_size=X_test.shape[1])
            # Find optimal decision threshold on val set, then apply to test.
            _, val_proba, val_y = predict_lstm(model, X_val, val_df)
            threshold = find_best_threshold(val_proba, val_y)
            log.info("LSTM  optimal threshold on val: %.2f", threshold)

            _, proba, y_lstm = predict_lstm(model, X_test, test_df)
            preds = (proba >= threshold).astype(int)
            y_test_for_metrics = y_lstm
            proba_for_metrics  = proba
        else:
            continue

        metrics = compute_metrics(y_test_for_metrics, preds, proba_for_metrics)

        # --- Per-ticker Sharpe, then average ---
        sharpe_values = []
        unique_tickers = sorted(set(test_tickers))

        if model_type == "lstm":
            # For LSTM, predictions are already per-ticker sequences (concatenated).
            # Reconstruct per-ticker slices by counting sequences per ticker.
            from .train import SEQUENCE_LENGTH as SEQ_LEN
            lstm_ticker_preds: dict[str, np.ndarray] = {}
            offset = 0
            for ticker in sorted(unique_tickers):
                n_rows = int((test_tickers == ticker).sum())
                n_seqs = n_rows - SEQ_LEN + 1
                if n_seqs > 0:
                    lstm_ticker_preds[ticker] = preds[offset : offset + n_seqs]
                    offset += n_seqs

        for ticker in unique_tickers:
            if ticker not in ticker_close:
                continue

            if model_type == "lstm":
                ticker_preds = lstm_ticker_preds.get(ticker)
                if ticker_preds is None:
                    continue
                close = ticker_close[ticker].iloc[SEQ_LEN - 1:]
            else:
                mask = test_tickers == ticker
                ticker_preds = preds[mask]
                close = ticker_close[ticker]

            if len(ticker_preds) == len(close):
                s = sharpe_from_signals(ticker_preds, close)
                sharpe_values.append(s)

        sharpe = float(np.mean(sharpe_values)) if sharpe_values else float("nan")

        metrics["sharpe"] = round(sharpe, 3)
        metrics["model"] = model_type
        results.append(metrics)

        log.info(
            "%s  roc_auc=%.4f  f1=%.4f  sharpe=%.3f",
            model_type, metrics["roc_auc"], metrics["f1"], sharpe,
        )

    if not results:
        log.error("No results to display. Did train.py complete successfully?")
        return

    # --- Buy-and-hold baseline (SPY as market proxy) ---
    bh_sharpe = buy_and_hold_sharpe(ticker_close["SPY"])

    # --- Print table ---
    result_df = pd.DataFrame(results).set_index("model")
    result_df = result_df[["accuracy", "precision", "recall", "f1", "roc_auc", "sharpe"]]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — TEST SET (2024)")
    print("=" * 70)
    print(result_df.to_string())
    print("-" * 70)
    print(f"Buy-and-hold Sharpe (baseline): {bh_sharpe:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
