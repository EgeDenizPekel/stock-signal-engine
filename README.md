# Stock Signal Engine

An end-to-end ML system that ingests real stock market data, engineers predictive features, trains and tracks multiple models, and serves buy/hold signals through a REST API with a React dashboard.

The emphasis is on **ML system design** — not just a model in a notebook, but a production-style pipeline with proper data handling, experiment tracking, and a deployable API.

---

## Architecture

```
yfinance API
     │
     ▼
src/data/ingest.py          Download OHLCV data → data/raw/<TICKER>.csv
     │
     ▼
src/data/features.py        Feature engineering → data/processed/<TICKER>.csv
     │
     ▼
src/models/train.py         Train 4 model families, log all runs to MLflow
     │
     ▼
src/models/evaluate.py      Metrics + backtested Sharpe on held-out test set
     │
     ▼
api/main.py (FastAPI)       Serve signals, history, and model performance
     │
     ▼
frontend/ (React + Vite)    Dashboard — signal cards, price chart, model comparison [Phase 4]
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | yfinance, pandas, ta |
| ML | scikit-learn, XGBoost, PyTorch |
| Experiment tracking | MLflow |
| API | FastAPI, uvicorn, Pydantic |
| Frontend | React, Vite, Tailwind CSS, recharts |
| Infrastructure | Docker, docker-compose |

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# 1. Download OHLCV data for all tickers (skips if already cached)
python -m src.data.ingest

# 2. Feature engineering
python -m src.data.features

# 3. Train all models (logs to MLflow)
python -m src.models.train

# 4. Evaluate on held-out test set
python -m src.models.evaluate

# 5. Inspect all runs visually
mlflow ui

# 6. Export best model artifacts to models/
python -m src.models.export

# 7. Start the API
uvicorn api.main:app --reload
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server status and loaded model names |
| GET | `/signal/{ticker}?model=` | Current BUY/HOLD signal for a ticker |
| GET | `/history/{ticker}?days=&model=` | OHLCV + signals for last N trading days (20–365) |
| GET | `/models/performance` | Test-set metrics for all 4 models |

`model` parameter accepts: `logistic_regression`, `random_forest`, `xgboost`, `lstm` (default: `logistic_regression`)

---

## Tickers

AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, SPY

Training window: 2018–2022 · Validation: 2023 · Test: 2024

---

## Model Results (Test Set — 2024)

| Model | ROC-AUC | F1 | Sharpe |
|---|---|---|---|
| Logistic Regression | 0.615 | 0.327 | 1.050 |
| LSTM | 0.611 | 0.522 | 1.233 |
| XGBoost | 0.583 | 0.264 | 0.868 |
| Random Forest | 0.596 | 0.210 | 0.614 |
| Buy-and-hold baseline (SPY) | — | — | 1.913 |

### A note on predictive power

These models have modest predictive signal — ROC-AUC of 0.61 is meaningfully above random (0.5) but far from strong. This is expected. Stock price direction from public OHLCV data and standard technical indicators is one of the hardest prediction problems in existence; academic literature consistently reports similar ranges for this feature set.

**The point of this project is not to beat the market.** It is to demonstrate the full engineering discipline of an ML system:

- Lookahead bias prevention — all features use `.shift(1)` before joining with the target
- Strict chronological train/val/test splits — no random shuffling of time-series data
- Scaler fit on training data only, applied to val and test
- Class imbalance handled with `class_weight='balanced'`
- Model selection by ROC-AUC on the validation set
- Decision threshold tuned on validation set (LSTM), never on test
- Backtesting with Sharpe ratio, not just classification accuracy
- Full experiment tracking via MLflow across all model families and hyperparameter sweeps

---

## Project Structure

```
stock-signal-engine/
├── data/
│   ├── raw/                  OHLCV CSVs from yfinance (gitignored)
│   └── processed/            Feature-engineered CSVs (gitignored)
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   └── features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── lstm.py
│   └── backtest/
│       └── strategy.py
├── api/
│   ├── main.py               App entry point; loads all model artifacts at startup
│   ├── schemas.py            Pydantic response models
│   └── routers/
│       ├── signal.py         GET /signal/{ticker}, GET /history/{ticker}
│       └── performance.py    GET /models/performance
├── models/                   Exported model artifacts (ships with repo)
├── tests/
│   └── test_api.py           Integration tests (20 tests; requires server on :8000)
├── mlruns/                   MLflow tracking (gitignored)
└── requirements.txt
```
