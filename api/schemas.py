from pydantic import BaseModel


class SignalResponse(BaseModel):
    ticker: str
    model: str
    signal: str          # "BUY" or "HOLD"
    probability: float
    generated_at: str


class HistoryPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    signal: str
    probability: float


class HistoryResponse(BaseModel):
    ticker: str
    model: str
    days: int
    data: list[HistoryPoint]


class ModelMetrics(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    sharpe: float


class PerformanceResponse(BaseModel):
    models: list[ModelMetrics]
