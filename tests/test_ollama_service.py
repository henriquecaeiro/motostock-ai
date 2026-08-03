"""Ollama service unit tests."""

import asyncio
import json
import math

import httpx
import pytest

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaService,
    OllamaTimeoutError,
)


def _install_mock_transport(
    ollama: OllamaService,
    handler,
) -> None:
    ollama._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=ollama.base_url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )


def _successful_embed_response(embeddings: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"embeddings": embeddings},
        headers={"Content-Type": "application/json"},
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
            _install_mock_transport(ollama, handler)

            with pytest.raises(OllamaModelUnavailableError) as exc_info:
                await ollama.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=missing_model,
                )

            assert exc_info.value.model == missing_model

    asyncio.run(run_test())


def test_embed_accepts_single_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return _successful_embed_response([[0.1, 0.2, 0.3]])

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            result = await ollama.embed("Demand forecasting estimates future sales.")

        assert result == [[0.1, 0.2, 0.3]]

    asyncio.run(run_test())


def test_embed_accepts_string_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _successful_embed_response(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            result = await ollama.embed(
                [
                    "Demand forecasting estimates future sales.",
                    "Safety stock protects against uncertainty.",
                ]
            )

        assert result == [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

    asyncio.run(run_test())


def test_embed_strips_external_whitespace() -> None:
    captured_payload: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return _successful_embed_response([[0.1]])

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            await ollama.embed("   trimmed text   ")

        assert captured_payload is not None
        assert captured_payload["input"] == ["trimmed text"]

    asyncio.run(run_test())


def test_embed_converts_integer_values_to_float() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _successful_embed_response([[1, 2, 3]])

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            result = await ollama.embed("numeric conversion")

        assert result == [[1.0, 2.0, 3.0]]

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "inputs",
    [
        [],
        "",
        "   ",
        ["valid", ""],
        ["valid", "   "],
        ["valid", 123],
        123,
    ],
)
def test_embed_rejects_invalid_inputs(inputs) -> None:
    async def run_test() -> None:
        async with OllamaService() as ollama:
            with pytest.raises((ValueError, TypeError)):
                await ollama.embed(inputs)

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "response_payload",
    [
        {},
        {"embeddings": None},
        {"embeddings": []},
        {"embeddings": [[0.1], [0.2]]},
        {"embeddings": [[]]},
        {"embeddings": [[0.1, "invalid"]]},
        {"embeddings": [[0.1, True]]},
        {"embeddings": [[0.1, math.nan]]},
        {"embeddings": [[0.1, math.inf]]},
        {"embeddings": [[0.1, -math.inf]]},
        {"embeddings": [[0.1, 0.2], [0.3]]},
        {"embeddings": "invalid"},
    ],
)
def test_embed_rejects_invalid_responses(response_payload) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=json.dumps(
                response_payload,
                allow_nan=True,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            with pytest.raises(OllamaInvalidResponseError):
                await ollama.embed("single input")

    asyncio.run(run_test())


def test_embed_maps_http_404_to_model_unavailable_error() -> None:
    missing_model = "qwen3-embedding:missing"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={"error": f"model '{missing_model}' not found"},
            headers={"Content-Type": "application/json"},
        )

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            with pytest.raises(OllamaModelUnavailableError) as exc_info:
                await ollama.embed(
                    ["text"],
                    model=missing_model,
                )

            assert exc_info.value.model == missing_model

    asyncio.run(run_test())


def test_embed_timeout_raises_ollama_timeout_error() -> None:
    async def failing_request(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    async def run_test() -> None:
        async with OllamaService() as ollama:
            ollama._client.request = failing_request  # type: ignore[method-assign]
            with pytest.raises(OllamaTimeoutError):
                await ollama.embed("timed out request")

    asyncio.run(run_test())


def test_embed_connection_error_raises_ollama_connection_error() -> None:
    async def failing_request(*args, **kwargs):
        raise httpx.ConnectError("connection failed")

    async def run_test() -> None:
        async with OllamaService() as ollama:
            ollama._client.request = failing_request  # type: ignore[method-assign]
            with pytest.raises(OllamaConnectionError):
                await ollama.embed("offline request")

    asyncio.run(run_test())


def test_embed_non_404_http_error_raises_ollama_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={"error": "internal server error"},
            headers={"Content-Type": "application/json"},
        )

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            with pytest.raises(OllamaRequestError):
                await ollama.embed("server error")

    asyncio.run(run_test())


def test_embed_uses_single_request_for_multiple_inputs() -> None:
    request_count = 0
    captured_payload: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count, captured_payload
        request_count += 1
        captured_payload = json.loads(request.content.decode("utf-8"))
        return _successful_embed_response(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

    async def run_test() -> None:
        async with OllamaService() as ollama:
            _install_mock_transport(ollama, handler)
            result = await ollama.embed(
                [
                    "first text",
                    "second text",
                    "third text",
                ]
            )

        assert request_count == 1
        assert captured_payload is not None
        assert captured_payload["input"] == [
            "first text",
            "second text",
            "third text",
        ]
        assert len(result) == 3

    asyncio.run(run_test())


def test_embed_payload_uses_selected_model_and_keep_alive() -> None:
    captured_payload: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return _successful_embed_response([[0.1, 0.2]])

    async def run_test() -> None:
        async with OllamaService(
            embedding_model="custom-embedding:1b",
            keep_alive="10m",
        ) as ollama:
            _install_mock_transport(ollama, handler)
            await ollama.embed("payload check")

        assert captured_payload is not None
        assert captured_payload["model"] == "custom-embedding:1b"
        assert captured_payload["input"] == ["payload check"]
        assert captured_payload["keep_alive"] == "10m"

    asyncio.run(run_test())


def test_default_embedding_model_comes_from_constructor() -> None:
    service = OllamaService(embedding_model="constructor-embedding:1b")

    assert service.default_embedding_model == "constructor-embedding:1b"


def test_default_embedding_model_comes_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "env-embedding:1b")

    service = OllamaService()

    assert service.default_embedding_model == "env-embedding:1b"


def test_empty_embedding_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="embedding model cannot be empty"):
        OllamaService(embedding_model="   ")
