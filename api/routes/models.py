"""Explicit model registry and promotion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas.models import ModelListResponse

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List registered model versions",
)
def list_models(request: Request) -> dict:
    models = request.app.state.repository.list_model_versions()
    return {"count": len(models), "models": models}


@router.post(
    "/models/{version}/promote",
    summary="Explicitly promote a candidate model",
)
def promote_model(request: Request, version: str) -> dict:
    repository = request.app.state.repository
    if not hasattr(repository, "promote_model_version"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model promotion requires DATA_BACKEND=sqlite.",
        )
    try:
        result = repository.promote_model_version(version)
        request.app.state.model_service.load()
        return dict(result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/models/{version}/rollback",
    summary="Roll back a production model",
)
def rollback_model(request: Request, version: str) -> dict:
    repository = request.app.state.repository
    if not hasattr(repository, "rollback_model_version"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model rollback requires DATA_BACKEND=sqlite.",
        )
    try:
        result = repository.rollback_model_version(version)
        request.app.state.model_service.load()
        return dict(result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
