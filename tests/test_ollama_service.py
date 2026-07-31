"""Ollama service unit tests."""

import asyncio

import httpx
import pytest

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaModelUnavailableError,
    OllamaService,
    OllamaServiceError,
)


def test_ensure_model_available_passes_when_model_is_installed(monkeypatch):
    """Do not raise when the configured model is present."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_list_models(self):
        return ["qwen3:4b", "qwen3:8b"]

    monkeypatch.setattr(OllamaService, "list_models", mock_list_models)

    async def run_test() -> None:
        async with OllamaService() as ollama:
            await ollama.ensure_model_available()

    asyncio.run(run_test())


def test_ensure_model_available_raises_when_model_is_missing(monkeypatch):
    """Raise OllamaModelUnavailableError when the model is not installed."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_list_models(self):
        return ["qwen3:8b"]

    monkeypatch.setattr(OllamaService, "list_models", mock_list_models)

    async def run_test() -> None:
        async with OllamaService() as ollama:
            with pytest.raises(OllamaModelUnavailableError) as exc_info:
                await ollama.ensure_model_available()

            assert exc_info.value.model == "qwen3:4b"

    asyncio.run(run_test())


def test_ensure_model_available_propagates_connection_errors(monkeypatch):
    """Let connection errors bubble up from list_models."""

    async def mock_list_models(self):
        raise OllamaConnectionError("Could not connect to Ollama.")

    monkeypatch.setattr(OllamaService, "list_models", mock_list_models)

    async def run_test() -> None:
        async with OllamaService() as ollama:
            with pytest.raises(OllamaConnectionError):
                await ollama.ensure_model_available()

    asyncio.run(run_test())


def test_chat_maps_ollama_404_to_model_unavailable_error():
    """Map Ollama HTTP 404 responses to OllamaModelUnavailableError."""

    missing_model = "qwen3:missing"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            return httpx.Response(
                status_code=404,
                json={"error": f"model '{missing_model}' not found"},
                headers={"Content-Type": "application/json"},
            )

        raise AssertionError(f"Unexpected request path: {request.url.path}")

    async def run_test() -> None:
        async with OllamaService() as ollama:
            ollama._client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url=ollama.base_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            with pytest.raises(OllamaModelUnavailableError) as exc_info:
                await ollama.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=missing_model,
                )

            assert exc_info.value.model == missing_model

    asyncio.run(run_test())
