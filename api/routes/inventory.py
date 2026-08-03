"""Inventory snapshot ingestion endpoints for offline POS synchronization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.inventory import (
    InventorySnapshotBatchCreate,
    InventorySnapshotCreate,
    InventorySnapshotWriteResponse,
)

router = APIRouter(tags=["Inventory"])


def _write_snapshots(request: Request, records: list[dict]) -> dict:
    """Write through the repository while keeping validation errors safe."""

    try:
        return dict(request.app.state.repository.insert_inventory_snapshots(records))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/inventory/snapshots",
    response_model=InventorySnapshotWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert one inventory snapshot",
    description=(
        "Insert one timestamped, idempotent stock observation for a known "
        "MotoStock product."
    ),
)
def create_inventory_snapshot(
    request: Request,
    payload: InventorySnapshotCreate,
) -> dict:
    record = payload.model_dump(mode="json")
    if not record.get("external_id") and not record.get("idempotency_key"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide external_id or idempotency_key.",
        )

    result = _write_snapshots(request, [record])
    return {
        "message": "Inventory snapshot accepted.",
        **result,
    }


@router.post(
    "/inventory/snapshots/batch",
    response_model=InventorySnapshotWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert a batch of inventory snapshots",
    description=(
        "Insert up to 1000 timestamped inventory snapshots in one transaction."
    ),
)
def create_inventory_snapshots_batch(
    request: Request,
    payload: InventorySnapshotBatchCreate,
) -> dict:
    records = [snapshot.model_dump(mode="json") for snapshot in payload.snapshots]
    for record in records:
        if not record.get("external_id") and not record.get("idempotency_key"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Every snapshot must provide external_id or idempotency_key.",
            )

    result = _write_snapshots(request, records)
    return {
        "message": "Inventory snapshots batch processed.",
        **result,
    }
