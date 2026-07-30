"""AI assistant endpoints."""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from api.schemas.assistant import AssistantHealthResponse
from api.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
)

router = APIRouter(
    prefix="/assistant",
    tags=["Assistant"],
)


@router.get(
    "/health",
    response_model=AssistantHealthResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": AssistantHealthResponse,
            "description": (
                "Ollama is unavailable or the configured model is not installed."
            ),
        }
    },
)
async def assistant_health() -> AssistantHealthResponse | JSONResponse:
    """Check whether Ollama and the configured model are available."""

    async with OllamaService() as ollama:
        try:
            available_models = await ollama.list_models()
        except OllamaServiceError:
            response = AssistantHealthResponse(
                status="ollama_unavailable",
                ollama_available=False,
                model=ollama.default_model,
                model_available=False,
            )

            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(),
            )

        model_available = ollama.default_model in available_models

        if not model_available:
            response = AssistantHealthResponse(
                status="model_unavailable",
                ollama_available=True,
                model=ollama.default_model,
                model_available=False,
            )

            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(),
            )

        return AssistantHealthResponse(
            status="ok",
            ollama_available=True,
            model=ollama.default_model,
            model_available=True,
        )
