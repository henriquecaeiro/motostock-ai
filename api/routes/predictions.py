"""Demand prediction endpoint."""

from fastapi import APIRouter, Request

from api.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Forecast demand for one product",
    description=(
        "Generate a recursive multi-day demand forecast for a known product "
        "using the production XGBoost pipeline."
    ),
)
def predict_demand(request: Request, payload: PredictionRequest) -> dict:
    forecast_service = request.app.state.forecast_service
    return forecast_service.predict_product(
        product_name=payload.product_name,
        horizon_days=payload.horizon_days,
    )
