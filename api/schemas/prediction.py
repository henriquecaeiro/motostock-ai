"""Prediction request and response schemas."""

from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for product demand prediction."""

    product_name: str = Field(..., min_length=1, description="Known product name")
    horizon_days: int = Field(
        default=14,
        ge=1,
        le=30,
        description="Number of future days to forecast (1-30)",
    )


class DailyForecastItem(BaseModel):
    """One day of forecasted demand."""

    date: date
    forecast_raw: float
    forecast_non_negative: float


class PredictionResponse(BaseModel):
    """Forecast response for a single product."""

    product_name: str
    forecast_start_date: date
    forecast_end_date: date
    horizon_days: int
    forecasted_demand_raw: float
    forecasted_demand_units: int
    daily_forecast: list[DailyForecastItem]
