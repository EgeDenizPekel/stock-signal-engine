"""
API integration tests.

Requires the server to be running:
    uvicorn api.main:app --port 8000

Run with:
    pytest tests/test_api.py -v
"""

import pytest
import httpx

BASE = "http://127.0.0.1:8000"
VALID_TICKER = "AAPL"
INVALID_TICKER = "INVALIDXYZ"
ALL_MODELS = ["logistic_regression", "random_forest", "xgboost", "lstm"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health():
    r = httpx.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert len(data["models_loaded"]) == 4


# ---------------------------------------------------------------------------
# /signal/{ticker}
# ---------------------------------------------------------------------------


def test_signal_default_model():
    r = httpx.get(f"{BASE}/signal/{VALID_TICKER}")
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == VALID_TICKER
    assert data["model"] == "logistic_regression"
    assert data["signal"] in ("BUY", "HOLD")
    assert 0.0 <= data["probability"] <= 1.0
    assert "generated_at" in data


@pytest.mark.parametrize("model", ALL_MODELS)
def test_signal_all_models(model):
    r = httpx.get(f"{BASE}/signal/{VALID_TICKER}", params={"model": model})
    assert r.status_code == 200
    data = r.json()
    assert data["signal"] in ("BUY", "HOLD")
    assert data["model"] == model


def test_signal_invalid_ticker():
    r = httpx.get(f"{BASE}/signal/{INVALID_TICKER}", timeout=30)
    assert r.status_code == 404


def test_signal_invalid_model():
    r = httpx.get(f"{BASE}/signal/{VALID_TICKER}", params={"model": "not_a_model"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /history/{ticker}
# ---------------------------------------------------------------------------


def test_history_default():
    r = httpx.get(f"{BASE}/history/{VALID_TICKER}", timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == VALID_TICKER
    assert data["model"] == "logistic_regression"
    assert data["days"] == 90
    assert len(data["data"]) == 90


def test_history_custom_days():
    r = httpx.get(f"{BASE}/history/{VALID_TICKER}", params={"days": 30}, timeout=30)
    assert r.status_code == 200
    assert r.json()["days"] == 30
    assert len(r.json()["data"]) == 30


@pytest.mark.parametrize("model", ALL_MODELS)
def test_history_all_models(model):
    r = httpx.get(
        f"{BASE}/history/{VALID_TICKER}",
        params={"days": 30, "model": model},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 30
    for point in data["data"]:
        assert point["signal"] in ("BUY", "HOLD")
        assert 0.0 <= point["probability"] <= 1.0


def test_history_point_fields():
    r = httpx.get(f"{BASE}/history/{VALID_TICKER}", params={"days": 20}, timeout=30)
    assert r.status_code == 200
    point = r.json()["data"][0]
    for field in ("date", "open", "high", "low", "close", "volume", "signal", "probability"):
        assert field in point


def test_history_invalid_ticker():
    r = httpx.get(f"{BASE}/history/{INVALID_TICKER}", timeout=30)
    assert r.status_code == 404


def test_history_days_out_of_range():
    r = httpx.get(f"{BASE}/history/{VALID_TICKER}", params={"days": 5})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /models/performance
# ---------------------------------------------------------------------------


def test_performance():
    r = httpx.get(f"{BASE}/models/performance")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) == 4


def test_performance_fields():
    r = httpx.get(f"{BASE}/models/performance")
    model = r.json()["models"][0]
    for field in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "sharpe"):
        assert field in model


def test_performance_model_names():
    r = httpx.get(f"{BASE}/models/performance")
    names = {m["model"] for m in r.json()["models"]}
    assert names == {"logistic_regression", "random_forest", "xgboost", "lstm"}
