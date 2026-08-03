"""Schemas for the operational recommendation refresh job."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RefreshRequest(BaseModel):
    """Parameters for one bounded refresh."""

    model_config = ConfigDict(extra="forbid")

    horizon_days: int = Field(default=14, ge=1, le=30)


class RefreshResponse(BaseModel):
    """Status and summary for a completed refresh."""

    status: Literal["completed"]
    run_key: str
    selected_model: str
    horizon_days: int
    features_rows: int
    recommendations_count: int
    last_historical_date: datetime
    started_at: datetime
    finished_at: datetime
