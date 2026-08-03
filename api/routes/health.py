"""Health check endpoint."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Check API and model availability",
    description="Returns basic service status and whether the forecasting model is loaded.",
)
def health_check(request: Request) -> dict:
    model_service = request.app.state.model_service
    return {
        "status": "ok",
        "project": "MotoStock AI",
        "model_loaded": model_service.is_loaded,
    }
