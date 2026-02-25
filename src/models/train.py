"""
Unified training script for all model families.

Trains four model types against the processed feature data and logs every run
to a single MLflow experiment. After this script completes, run `mlflow ui`
to compare all runs visually.

Models trained
--------------
1. Logistic Regression  (sklearn)
2. Random Forest        (sklearn)
3. XGBoost              (xgboost) — small hyperparameter sweep, 3 runs
4. LSTM                 (PyTorch)

Split
-----
Train : 2018-01-01 → 2022-12-31
Val   : 2023-01-01 → 2023-12-31
Test  : 2024-01-01 → 2024-12-31  ← held out, only used in evaluate.py

Usage
-----
    python -m src.models.train
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from .lstm import LSTMClassifier, SequenceDataset, SEQUENCE_LENGTH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MLFLOW_EXPERIMENT = "stock-signal-engine"

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END   = "2023-12-31"
TEST_START = "2024-01-01"

# Features that should NOT be scaled (already-bounded small integers)
CALENDAR_COLS = ["day_of_week", "month", "quarter"]

# LSTM training hyperparameters
LSTM_EPOCHS     = 30
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 10  # early stopping patience (epochs without val improvement)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_all() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No processed CSVs found. Run features.py first.")
    df = pd.concat(
        [pd.read_csv(f, index_col="Date", parse_dates=True) for f in files]
    ).sort_index()
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All columns except ticker and target."""
    return [c for c in df.columns if c not in ("ticker", "target")]


def split(df: pd.DataFrame):
    train = df.loc[:TRAIN_END]
    val   = df.loc[VAL_START:VAL_END]
    test  = df.loc[TEST_START:]
    return train, val, test


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train continuous features only.
    Calendar columns are passed through unchanged.
    Returns scaled numpy arrays and the fitted scaler.
    """
    continuous_cols = [c for c in feature_cols if c not in CALENDAR_COLS]
    calendar_cols   = [c for c in feature_cols if c in CALENDAR_COLS]

    scaler = StandardScaler()
    scaler.fit(train[continuous_cols])

    def transform(split_df: pd.DataFrame) -> np.ndarray:
        scaled = scaler.transform(split_df[continuous_cols])
        calendar = split_df[calendar_cols].values
        return np.hstack([scaled, calendar])

    return (
        transform(train),
        transform(val),
        transform(test),
        scaler,
    )

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def val_roc_auc(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))

# ---------------------------------------------------------------------------
# LSTM sequence builder (per-ticker to avoid boundary crossings)
# ---------------------------------------------------------------------------


def build_lstm_sequences(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rolling-window sequences for the LSTM, keeping each ticker's
    time series isolated. Windows that would cross a ticker boundary
    are never created.

    Parameters
    ----------
    df      : DataFrame with a 'ticker' column (same rows as X_scaled / y).
    X_scaled: Scaled feature matrix of shape (n_rows, n_features).
    y       : Label vector of shape (n_rows,).

    Returns
    -------
    X_seq : (n_windows, SEQUENCE_LENGTH, n_features)
    y_seq : (n_windows,)
    """
    all_X, all_y = [], []
    tickers = df["ticker"].values

    for ticker in sorted(set(tickers)):
        mask = tickers == ticker
        X_t = X_scaled[mask]
        y_t = y[mask]

        for i in range(len(X_t) - SEQUENCE_LENGTH + 1):
            all_X.append(X_t[i : i + SEQUENCE_LENGTH])
            all_y.append(y_t[i + SEQUENCE_LENGTH - 1])

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def train_logistic(
    X_train, y_train, X_val, y_val, run_name: str = "logistic_regression"
) -> None:
    log.info("Training Logistic Regression ...")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "logistic_regression")
        params = dict(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        auc = val_roc_auc(model, X_val, y_val)
        mlflow.log_metric("val_roc_auc", auc)
        log.info("Logistic Regression  val_roc_auc=%.4f", auc)

        mlflow.sklearn.log_model(model, artifact_path="model")


def train_random_forest(
    X_train, y_train, X_val, y_val, run_name: str = "random_forest"
) -> None:
    log.info("Training Random Forest ...")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "random_forest")
        params = dict(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        auc = val_roc_auc(model, X_val, y_val)
        mlflow.log_metric("val_roc_auc", auc)
        log.info("Random Forest  val_roc_auc=%.4f", auc)

        mlflow.sklearn.log_model(model, artifact_path="model")


def train_xgboost(X_train, y_train, X_val, y_val) -> None:
    """Small hyperparameter sweep — each combo is a separate MLflow run."""
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    sweep = [
        dict(n_estimators=100, max_depth=4, learning_rate=0.1),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05),
        dict(n_estimators=300, max_depth=6, learning_rate=0.05),
    ]

    for i, hp in enumerate(sweep):
        run_name = f"xgboost_{i+1}"
        log.info("Training XGBoost run %d/%d  params=%s", i + 1, len(sweep), hp)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", "xgboost")
            params = {**hp, "scale_pos_weight": scale_pos_weight, "random_state": 42, "eval_metric": "auc"}
            mlflow.log_params(params)

            model = XGBClassifier(**params, verbosity=0)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            proba = model.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, proba))
            mlflow.log_metric("val_roc_auc", auc)
            log.info("XGBoost run %d  val_roc_auc=%.4f", i + 1, auc)

            mlflow.xgboost.log_model(model, artifact_path="model")


def train_lstm(
    X_train, y_train, X_val, y_val,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    run_name: str = "lstm",
) -> None:
    log.info("Training LSTM ...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("LSTM device: %s", device)

    # Build per-ticker sequences to avoid windows crossing ticker boundaries
    X_train_seq, y_train_seq = build_lstm_sequences(train_df, X_train, y_train)
    X_val_seq,   y_val_seq   = build_lstm_sequences(val_df,   X_val,   y_val)
    log.info(
        "LSTM sequences — train: %s  val: %s",
        X_train_seq.shape, X_val_seq.shape,
    )

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train_seq), torch.tensor(y_train_seq)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val_seq), torch.tensor(y_val_seq)
    )
    train_dl = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=LSTM_BATCH_SIZE, shuffle=False)

    n_features = X_train.shape[1]
    model = LSTMClassifier(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "lstm")
        mlflow.log_params(dict(
            sequence_length=SEQUENCE_LENGTH,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            lr=LSTM_LR,
            batch_size=LSTM_BATCH_SIZE,
            max_epochs=LSTM_EPOCHS,
        ))

        for epoch in range(1, LSTM_EPOCHS + 1):
            # --- train ---
            model.train()
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

            # --- validate ---
            model.eval()
            val_losses, all_proba, all_labels = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    proba = model(X_batch)
                    val_losses.append(criterion(proba, y_batch).item())
                    all_proba.extend(proba.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            val_loss = float(np.mean(val_losses))
            val_auc  = float(roc_auc_score(all_labels, all_proba))
            mlflow.log_metrics({"val_loss": val_loss, "val_roc_auc": val_auc}, step=epoch)

            if epoch % 5 == 0:
                log.info(
                    "LSTM  epoch %2d/%d  val_loss=%.4f  val_roc_auc=%.4f",
                    epoch, LSTM_EPOCHS, val_loss, val_auc,
                )

            # --- early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= LSTM_PATIENCE:
                    log.info("LSTM  early stopping at epoch %d", epoch)
                    break

        # Reload best weights before saving
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        all_proba, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(device)
                proba = model(X_batch)
                all_proba.extend(proba.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        final_auc = float(roc_auc_score(all_labels, all_proba))
        mlflow.log_metric("val_roc_auc_best", final_auc)
        log.info("LSTM  best val_roc_auc=%.4f", final_auc)

        mlflow.pytorch.log_model(model, artifact_path="model")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    log.info("Loading processed data ...")
    df = load_all()
    feature_cols = get_feature_cols(df)

    train_df, val_df, test_df = split(df)
    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    X_train, X_val, X_test, scaler = scale_features(
        train_df, val_df, test_df, feature_cols
    )
    y_train = train_df["target"].values
    y_val   = val_df["target"].values

    log.info("Feature matrix shape — train: %s  val: %s", X_train.shape, X_val.shape)

    train_logistic(X_train, y_train, X_val, y_val)
    train_random_forest(X_train, y_train, X_val, y_val)
    train_xgboost(X_train, y_train, X_val, y_val)
    train_lstm(X_train, y_train, X_val, y_val, train_df, val_df)

    log.info("All models trained. Run `mlflow ui` to compare runs.")


if __name__ == "__main__":
    main()
