"""
Download and cache OHLCV data from Yahoo Finance via yfinance.

Saves one CSV per ticker to data/raw/<TICKER>.csv.
Re-runs are skipped if the file already exists — delete the file to force re-download.

Usage:
    python src/data/ingest.py
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "SPY"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def download_ticker(ticker: str, out_dir: Path) -> Path:
    """
    Download OHLCV data for a single ticker and save to CSV.

    Skips the download if the file already exists.

    Returns the path to the saved CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.csv"

    if out_path.exists():
        log.info("%s  already cached, skipping", ticker)
        return out_path

    log.info("%s  downloading %s → %s ...", ticker, START_DATE, END_DATE)
    df: pd.DataFrame = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance may return a MultiIndex when auto_adjust=True on newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure index is named consistently
    df.index.name = "Date"

    df.to_csv(out_path)
    log.info("%s  saved %d rows → %s", ticker, len(df), out_path)
    return out_path


def ingest_all() -> None:
    """Download all configured tickers sequentially."""
    failed = []
    for ticker in TICKERS:
        try:
            download_ticker(ticker, RAW_DIR)
        except Exception as exc:
            log.error("%s  FAILED: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        raise RuntimeError(f"Ingestion failed for: {failed}")

    log.info("Ingestion complete. %d tickers in %s", len(TICKERS), RAW_DIR)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest_all()
