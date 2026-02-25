"""
Feature engineering pipeline.

Loads raw OHLCV CSVs from data/raw/, computes all technical indicator groups,
creates the binary classification target, drops NaN rows, and saves one
processed CSV per ticker to data/processed/.

Target variable
---------------
    target = 1  if forward_5d_return > TARGET_THRESHOLD (2%)
    target = 0  otherwise (flat or down)

    Forward return is computed without lookahead: we know today's close price
    and predict the return over the *next* 5 trading days. The feature values
    for a given row are computed using data available *before* that day's close
    (all rolling features use .shift(1) before joining with the target).

Usage:
    python src/data/features.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "SPY"]
TARGET_THRESHOLD = 0.02   # 2 % rise over 5 trading days → label = 1
FORWARD_DAYS = 5

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found for {ticker}. Run ingest.py first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    return df


def _build_target(close: pd.Series) -> pd.Series:
    """
    Binary label: 1 if price rises > TARGET_THRESHOLD in next FORWARD_DAYS days.

    Uses .shift(-FORWARD_DAYS) to look forward, which is valid because we only
    use this column as the label y — never as an input feature.
    """
    forward_return = close.shift(-FORWARD_DAYS) / close - 1
    return (forward_return > TARGET_THRESHOLD).astype(int)


def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["sma_20"] = ta.trend.sma_indicator(close, window=20)
    feats["sma_50"] = ta.trend.sma_indicator(close, window=50)
    feats["ema_12"] = ta.trend.ema_indicator(close, window=12)
    feats["ema_26"] = ta.trend.ema_indicator(close, window=26)
    # Price position relative to moving averages (ratio, scale-invariant)
    feats["price_to_sma20"] = close / feats["sma_20"]
    feats["price_to_sma50"] = close / feats["sma_50"]
    return feats


def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["rsi_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    feats["macd"] = macd.macd()
    feats["macd_signal"] = macd.macd_signal()
    feats["macd_hist"] = macd.macd_diff()
    return feats


def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    feats = pd.DataFrame(index=df.index)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    feats["bb_upper"] = bb.bollinger_hband()
    feats["bb_mid"] = bb.bollinger_mavg()
    feats["bb_lower"] = bb.bollinger_lband()
    feats["bb_width"] = (feats["bb_upper"] - feats["bb_lower"]) / feats["bb_mid"]
    feats["bb_pct"] = bb.bollinger_pband()  # (price - lower) / (upper - lower)
    feats["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)
    return feats


def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    volume = df["volume"]
    feats = pd.DataFrame(index=df.index)
    feats["obv"] = ta.volume.on_balance_volume(close, volume)
    feats["volume_sma_20"] = ta.trend.sma_indicator(volume, window=20)
    feats["volume_ratio"] = volume / feats["volume_sma_20"]
    return feats


def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["return_1d"] = close.pct_change(1)
    feats["return_5d"] = close.pct_change(5)
    feats["return_10d"] = close.pct_change(10)
    feats["return_20d"] = close.pct_change(20)
    feats["log_return_1d"] = np.log(close / close.shift(1))
    return feats


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["day_of_week"] = df.index.dayofweek          # 0=Mon … 4=Fri
    feats["month"] = df.index.month
    feats["quarter"] = df.index.quarter
    return feats


def _assemble_features(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Core feature computation on a raw OHLCV DataFrame.
    Shared by the batch pipeline and the live API.

    Applies shift(1) to all features so that features[t] represent
    information available before day t's close.

    Does NOT include the target column — callers add it if needed.
    Does NOT dropna — callers decide what to drop.
    """
    feature_blocks = [
        _trend_features(raw_df),
        _momentum_features(raw_df),
        _volatility_features(raw_df),
        _volume_features(raw_df),
        _price_features(raw_df),
        _calendar_features(raw_df),
    ]
    features = pd.concat(feature_blocks, axis=1)
    features = features.shift(1)

    out = pd.DataFrame(index=raw_df.index)
    out["ticker"] = ticker
    out = pd.concat([out, features], axis=1)
    return out


def build_features(ticker: str) -> pd.DataFrame:
    """
    Full feature engineering pipeline for one ticker (batch use).

    Returns a cleaned DataFrame with no NaN rows and a target column.
    Close price is not included — reconstruct from data/raw/ when needed.
    """
    df = _load_raw(ticker)
    target = _build_target(df["close"])
    out = _assemble_features(df, ticker)
    out["target"] = target

    n_before = len(out)
    out = out.dropna()
    log.info(
        "%s  features built: %d rows (%d dropped for NaN)",
        ticker, len(out), n_before - len(out),
    )
    return out


def compute_features_for_api(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Compute features from a live-fetched OHLCV DataFrame for API use.

    Expects raw_df to have lowercase OHLCV column names and a DatetimeIndex.
    Returns feature rows with NaN rows dropped. No target column included.
    Fetch at least 100 days of raw data to ensure rolling windows are populated.
    """
    raw_df = raw_df.copy()
    raw_df.columns = [c.lower() for c in raw_df.columns]
    raw_df = raw_df.sort_index()

    out = _assemble_features(raw_df, ticker)
    # Keep OHLCV alongside features for the history response
    for col in ["open", "high", "low", "close", "volume"]:
        if col in raw_df.columns:
            out[col] = raw_df[col]

    out = out.dropna(subset=[c for c in out.columns if c not in ("ticker", "open", "high", "low", "close", "volume")])
    return out


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def process_all() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for ticker in TICKERS:
        try:
            df = build_features(ticker)
            out_path = PROCESSED_DIR / f"{ticker}.csv"
            df.to_csv(out_path)
            log.info("%s  saved → %s", ticker, out_path)
        except Exception as exc:
            log.error("%s  FAILED: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        raise RuntimeError(f"Feature engineering failed for: {failed}")

    log.info("Feature engineering complete. %d tickers in %s", len(TICKERS), PROCESSED_DIR)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_all()
