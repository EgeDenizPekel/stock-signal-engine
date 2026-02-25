from fastapi import APIRouter
from api.schemas import PerformanceResponse, ModelMetrics
from api.main import registry

router = APIRouter()


@router.get("/models/performance", response_model=PerformanceResponse)
def get_performance():
    """Return test-set metrics for all trained models."""
    models = [ModelMetrics(**m) for m in registry["metrics"]]
    return PerformanceResponse(models=models)
