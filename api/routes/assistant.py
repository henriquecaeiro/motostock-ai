"""AI assistant endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.prompts import load_system_prompt
from api.schemas.assistant import (
    AssistantErrorResponse,
    AssistantHealthResponse,
    AssistantRequest,
    AssistantResponse,
)
from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaService,
    OllamaServiceError,
    OllamaTimeoutError,
)

logger = logging.getLogger(__name__)

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
async def assistant_health(
    request: Request,
) -> AssistantHealthResponse | JSONResponse:
    """Check whether Ollama and the configured model are available."""

    ollama: OllamaService = request.app.state.ollama_service

    try:
        available_models = await ollama.list_models()
    except OllamaServiceError:
        logger.warning("Assistant health check failed because Ollama is unavailable.")

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
        logger.warning(
            "Assistant health check failed because the configured model "
            "is unavailable: %s",
            ollama.default_model,
        )

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
            "model": AssistantErrorResponse,
            "description": "Ollama is unavailable or the configured model is missing.",
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "model": AssistantErrorResponse,
            "description": "The language model request timed out.",
        },
        status.HTTP_502_BAD_GATEWAY: {
            "model": AssistantErrorResponse,
            "description": "The language model returned an invalid or unexpected response.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": AssistantErrorResponse,
            "description": "The assistant system prompt could not be loaded.",
        },
    },
)
async def assistant_chat(
    payload: AssistantRequest, request: Request
) -> AssistantResponse:
    """Send a user message to the MotoStockAI assistant."""

    ollama: OllamaService = request.app.state.ollama_service

    try:
        system_prompt = load_system_prompt()
    except RuntimeError as exc:
        logger.error("Assistant configuration could not be loaded.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The assistant configuration could not be loaded.",
        ) from exc

    try:
        answer = await ollama.ask(
            prompt=payload.message,
            system_prompt=system_prompt,
            options={
                "temperature": 0.1,
            },
        )

    except OllamaModelUnavailableError as exc:
        logger.warning(
            "Assistant request failed because the configured model is unavailable: %s",
            exc.model,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configured Ollama model is unavailable: {exc.model}",
        ) from exc

    except OllamaConnectionError:
        logger.warning("Assistant request failed because Ollama is unavailable.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable.",
        )

    except OllamaTimeoutError:
        logger.warning(
            "Assistant request timed out for model: %s",
            ollama.default_model,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The language model request timed out.",
        )

    except OllamaInvalidResponseError:
        logger.error(
            "Assistant request failed because Ollama returned an invalid response."
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model returned an invalid response.",
        )

    except OllamaRequestError:
        logger.error(
            "Assistant request failed because the language model service "
            "returned an unexpected error."
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model service returned an unexpected error.",
        )

    return AssistantResponse(
        answer=answer,
        model=ollama.default_model,
    )
