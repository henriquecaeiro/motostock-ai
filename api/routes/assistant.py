"""AI assistant endpoints."""

"""AI assistant endpoints."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.prompts import load_system_prompt
from api.schemas.assistant import (
    AssistantHealthResponse,
    AssistantRequest,
    AssistantResponse,
)
from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaRequestError,
    OllamaService,
    OllamaServiceError,
    OllamaTimeoutError,
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


@router.post(
    "/chat",
    response_model=AssistantResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The Ollama server is unavailable.",
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "The language model took too long to respond.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Ollama returned an invalid or unsuccessful response.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "The assistant system prompt could not be loaded.",
        },
    },
)
async def assistant_chat(request: AssistantRequest) -> AssistantResponse:
    """Send a user message to the MotoStockAI assistant."""

    try:
        system_prompt = load_system_prompt()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The assistant configuration could not be loaded.",
        ) from exc

    async with OllamaService() as ollama:
        try:
            answer = await ollama.ask(
                prompt=request.message,
                system_prompt=system_prompt,
                options={
                    "temperature": 0.1,
                },
            )

        except OllamaTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The AI assistant took too long to respond",
            ) from exc

        except OllamaConnectionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The AI assistant is temporarily unavailable "
                    "because Ollama could not be reached."
                ),
            ) from exc

        except (
            OllamaRequestError,
            OllamaInvalidResponseError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "The AI assistant received an invalid response "
                    "from the language model."
                ),
            ) from exc

        return AssistantResponse(
            answer=answer,
            model=ollama.default_model,
        )
