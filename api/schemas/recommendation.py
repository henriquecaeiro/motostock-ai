"""Recommendation response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


StockStatus = Literal["critical", "warning", "healthy", "overstock"]


class RecommendationItem(BaseModel):
    """One stock recommendation row."""

    product_name: str
    forecast_horizon_days: int
    forecasted_demand_raw: float
    forecasted_demand_non_negative: float
    forecasted_demand_units: int
    current_stock: int
    safety_stock: int
    required_stock: int
    recommended_purchase_quantity: int
    supplier_lead_time_days: int
    stock_status: StockStatus
    priority_score: int


class RecommendationsResponse(BaseModel):
    """Collection of stock recommendations."""

    generated_at: datetime
    selected_model: str
    horizon_days: int
    count: int
    recommendations: list[RecommendationItem]


class RecommendationsSummaryResponse(BaseModel):
    """Aggregated counts from the current recommendation result."""

    selected_model: str
    forecast_horizon_days: int
    total_products: int
    critical_products: int
    warning_products: int
    healthy_products: int
    overstock_products: int
    total_recommended_purchase_units: int
    highest_priority_product: str | None = Field(
        default=None,
        description="Product with the highest priority score among recommendations",
    )
