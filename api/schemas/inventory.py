"""Validation models for POS inventory snapshot ingestion."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InventorySnapshotCreate(BaseModel):
    """One timestamped, idempotent stock observation."""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1, max_length=200)
    quantity_on_hand: float = Field(ge=0)
    supplier_lead_time_days: int = Field(ge=1)
    observed_at: datetime
    external_id: str | None = Field(default=None, min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        """Require an explicit timezone so event ordering is unambiguous."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset.")
        return value

    @field_validator("external_id", "idempotency_key")
    @classmethod
    def normalize_event_key(cls, value: str | None) -> str | None:
        """Reject whitespace-only identifiers while preserving stable values."""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Event identifiers cannot be blank.")
        return normalized


class InventorySnapshotBatchCreate(BaseModel):
    """Bounded batch of inventory observations."""

    model_config = ConfigDict(extra="forbid")

    snapshots: list[InventorySnapshotCreate] = Field(min_length=1, max_length=1000)


class InventorySnapshotWriteResponse(BaseModel):
    """Result of an idempotent inventory snapshot write."""

    message: str
    inserted: int
    skipped: int
    updated: int
    snapshot_ids: list[int]
