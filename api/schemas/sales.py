"""Validation models for operational sales ingestion."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class SaleCreate(BaseModel):
    """One positive operational sale."""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1, max_length=200)
    sale_date: date
    quantity_sold: float = Field(gt=0)
    unit_price_brl: float = Field(gt=0)
    total_revenue_brl: float | None = Field(default=None, ge=0)
    estimated_profit_brl: float | None = None
    discount_pct: float | None = Field(default=None, ge=0, le=100)
    current_stock_snapshot: float | None = Field(default=None, ge=0)
    supplier_lead_time_days: int | None = Field(default=None, ge=1)
    weather_condition: str | None = Field(default=None, max_length=100)
    external_id: str | None = Field(default=None, min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class SalesBatchCreate(BaseModel):
    """Bounded batch of sales inserted in one transaction."""

    model_config = ConfigDict(extra="forbid")

    sales: list[SaleCreate] = Field(min_length=1, max_length=1000)


class SalesWriteResponse(BaseModel):
    """Result of an idempotent sales write."""

    message: str
    inserted: int
    skipped: int
    sale_ids: list[int]

