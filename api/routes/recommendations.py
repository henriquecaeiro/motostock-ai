"""Stock recommendation endpoints."""

from fastapi import APIRouter, HTTPException, Query, Request, status

from api.schemas.recommendation import (
    RecommendationsResponse,
    RecommendationsSummaryResponse,
)

router = APIRouter(tags=["Recommendations"])


@router.get(
    "/recommendations/latest",
    response_model=RecommendationsResponse,
    summary="Read the latest persisted recommendations",
    description=(
        "Return the most recent completed recommendation run stored by the "
        "configured repository."
    ),
)
def get_latest_recommendations(
    request: Request,
    horizon_days: int | None = Query(default=None, ge=1, le=30),
) -> dict:
    repository = request.app.state.repository
    payload = repository.get_latest_recommendations(horizon_days=horizon_days)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No persisted recommendation run is available.",
        )
    return dict(payload)


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    summary="Generate stock recommendations",
    description=(
        "Forecast demand for all products and apply stock recommendation rules "
        "using current stock and supplier lead time."
    ),
)
def get_recommendations(
    request: Request,
    horizon_days: int = Query(
        default=14,
        ge=1,
        le=30,
        description="Forecast horizon in days (1-30)",
    ),
    stock_status: str | None = Query(
        default=None,
        description="Optional stock status filter",
    ),
    product_name: str | None = Query(
        default=None,
        description="Optional product name filter",
    ),
) -> dict:
    recommendation_service = request.app.state.recommendation_service
    return recommendation_service.get_recommendations(
        horizon_days=horizon_days,
        stock_status=stock_status,
        product_name=product_name,
    )


@router.get(
    "/recommendations/summary",
    response_model=RecommendationsSummaryResponse,
    summary="Summarize stock recommendations",
    description="Return aggregate counts derived from the current recommendation result.",
)
def get_recommendations_summary(
    request: Request,
    horizon_days: int = Query(
        default=14,
        ge=1,
        le=30,
        description="Forecast horizon in days (1-30)",
    ),
) -> dict:
    recommendation_service = request.app.state.recommendation_service
    return recommendation_service.get_summary(horizon_days=horizon_days)
