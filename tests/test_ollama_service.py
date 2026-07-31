"""Ollama service unit tests."""

import asyncio

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
            with pytest.raises(OllamaModelUnavailableError):
                await ollama.ensure_model_available()

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


async def main() -> None:
    async with OllamaService() as ollama:
        available = await ollama.is_available()

        print(f"Ollama available: {available}")

        if not available:
            print("Start Ollama before running this test.")
            return

        models = await ollama.list_models()

        print(f"Installed models: {models}")

        if not await ollama.has_model():
            print(f"Required model is not installed: {ollama.default_model}")
            return

        answer = await ollama.ask(
            prompt=(
                "Explain safety stock in one short sentence. "
                "Return only the explanation."
            ),
            system_prompt=("You are a concise inventory assistant."),
            options={
                "temperature": 0.2,
            },
        )

        print("\nModel answer:")
        print(answer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except OllamaServiceError as exc:
        print(f"Ollama error: {exc}")
