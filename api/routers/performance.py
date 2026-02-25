from fastapi import APIRouter
from api.schemas import PerformanceResponse, ModelMetrics
from api.main import registry

router = APIRouter()


@router.get("/models/performance", response_model=PerformanceResponse)
def get_performance():
    """Return test-set and val-set metrics for all trained models."""
    data = registry["metrics"]
    models = [ModelMetrics(**m) for m in data["test"]]
    val_models = [ModelMetrics(**m) for m in data["val"]]
    return PerformanceResponse(models=models, val_models=val_models)
