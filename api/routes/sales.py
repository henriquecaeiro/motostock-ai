"""Operational sales ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status

from api.schemas.sales import SaleCreate, SalesBatchCreate, SalesWriteResponse

router = APIRouter(tags=["Sales"])


def _write_sales(request: Request, records: list[dict]) -> dict:
    """Write through the repository and translate validation failures to 422."""

    try:
        return dict(request.app.state.repository.insert_sales(records))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/sales",
    response_model=SalesWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert one operational sale",
    description=(
        "Insert a validated sale using an external_id, idempotency_key or "
        "the Idempotency-Key header."
    ),
)
def create_sale(
    request: Request,
    payload: SaleCreate,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    record = payload.model_dump(mode="json")
    if not record.get("idempotency_key") and idempotency_header:
        record["idempotency_key"] = idempotency_header.strip()
    if not record.get("idempotency_key") and not record.get("external_id"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide external_id, idempotency_key or Idempotency-Key header.",
        )

    result = _write_sales(request, [record])
    return {
        "message": "Sale accepted." if result["inserted"] else "Sale already exists.",
        **result,
    }


@router.post(
    "/sales/batch",
    response_model=SalesWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert a batch of operational sales",
    description="Insert up to 1000 sales in one transaction with idempotent keys.",
)
def create_sales_batch(
    request: Request,
    payload: SalesBatchCreate,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    records = [sale.model_dump(mode="json") for sale in payload.sales]
    for index, record in enumerate(records):
        if not record.get("idempotency_key") and idempotency_header:
            record["idempotency_key"] = f"{idempotency_header.strip()}:{index}"
        if not record.get("idempotency_key") and not record.get("external_id"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Every sale must provide external_id or idempotency_key; "
                    "the Idempotency-Key header can be used for the whole batch."
                ),
            )

    result = _write_sales(request, records)
    return {
        "message": "Sales batch processed.",
        **result,
    }
