"""Administrative model registry schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class ModelVersion(BaseModel):
    version: str
    model_name: str
    artifact_path: str
    artifact_sha256: str
    backup_artifact_path: str | None = None
    training_started_at: datetime | None = None
    training_finished_at: datetime | None = None
    data_start_date: str | None = None
    data_end_date: str | None = None
    feature_columns: list[str]
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    seed: int
    parent_version: str | None = None
    status: Literal["candidate", "production", "active", "rejected", "archived"]
    promoted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    count: int
    models: list[ModelVersion]
