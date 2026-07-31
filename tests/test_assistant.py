"""Assistant endpoint tests."""

import logging

import pytest

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaService,
    OllamaTimeoutError,
)


async def _mock_ensure_model_available(self, model=None):
    """No-op stand-in so chat tests do not call the real Ollama API."""

    return None


@pytest.fixture(autouse=True)
def _patch_ensure_model_available(monkeypatch):
    """Skip model availability checks unless a test overrides this behavior."""

    monkeypatch.setattr(
        OllamaService,
        "ensure_model_available",
        _mock_ensure_model_available,
    )


def test_assistant_health_returns_ok(client, monkeypatch):
    """Return HTTP 200 when Ollama and the configured model are available."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_list_models(self):
        return ["qwen3:4b"]

    monkeypatch.setattr(
        OllamaService,
        "list_models",
        mock_list_models,
    )

    response = client.get("/assistant/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ollama_available": True,
        "model": "qwen3:4b",
        "model_available": True,
    }


def test_assistant_health_returns_503_when_model_is_missing(client, monkeypatch):
    """Return HTTP 503 when Ollama is available but the model is missing."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_list_models(self):
        return ["qwen3:8b"]

    monkeypatch.setattr(
        OllamaService,
        "list_models",
        mock_list_models,
    )

    response = client.get("/assistant/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "model_unavailable",
        "ollama_available": True,
        "model": "qwen3:4b",
        "model_available": False,
    }


def test_assistant_health_returns_503_when_ollama_is_unavailable(
    client,
    monkeypatch,
):
    """Return HTTP 503 without exposing an exception when Ollama is offline."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_list_models(self):
        raise OllamaConnectionError("Could not connect to Ollama.")

    monkeypatch.setattr(
        OllamaService,
        "list_models",
        mock_list_models,
    )

    response = client.get("/assistant/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "ollama_unavailable",
        "ollama_available": False,
        "model": "qwen3:4b",
        "model_available": False,
    }


def test_assistant_chat_returns_answer(
    client,
    monkeypatch,
):
    """Return the assistant answer using the configured model."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        assert prompt == (
            "Explain the difference between demand forecasting "
            "and stock recommendation."
        )
        assert system_prompt
        assert options == {"temperature": 0.1}

        return (
            "Demand forecasting estimates future sales, while stock "
            "recommendation calculates replenishment needs."
        )

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": (
                "Explain the difference between demand forecasting "
                "and stock recommendation."
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Demand forecasting estimates future sales, while stock "
            "recommendation calculates replenishment needs."
        ),
        "model": "qwen3:4b",
        "tools_used": [],
        "sources": [],
    }


def test_assistant_chat_removes_external_whitespace(
    client,
    monkeypatch,
):
    """Remove unnecessary whitespace before sending the prompt."""

    received_prompt = None

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        nonlocal received_prompt
        received_prompt = prompt

        return "Safety stock is extra inventory."

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "   What is safety stock?   ",
        },
    )

    assert response.status_code == 200
    assert received_prompt == "What is safety stock?"


def test_assistant_chat_rejects_empty_message(client):
    """Reject a message containing only whitespace."""

    response = client.post(
        "/assistant/chat",
        json={
            "message": "     ",
        },
    )

    assert response.status_code == 422


def test_assistant_chat_rejects_message_above_limit(client):
    """Reject a message longer than 2,000 characters."""

    response = client.post(
        "/assistant/chat",
        json={
            "message": "a" * 2001,
        },
    )

    assert response.status_code == 422


def test_assistant_chat_returns_503_when_ollama_is_unavailable(
    client,
    monkeypatch,
):
    """Return HTTP 503 when the Ollama server cannot be reached."""

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaConnectionError("Could not connect to Ollama.")

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Ollama service is unavailable.",
    }


def test_assistant_chat_returns_503_when_model_is_unavailable(
    client,
    monkeypatch,
):
    """Return HTTP 503 when the configured model is not installed."""

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")

    async def mock_ensure_model_available(self, model=None):
        raise OllamaModelUnavailableError(
            "Configured Ollama model is not installed: qwen3:4b"
        )

    monkeypatch.setattr(
        OllamaService,
        "ensure_model_available",
        mock_ensure_model_available,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Configured Ollama model is unavailable: qwen3:4b",
    }


def test_assistant_chat_returns_504_on_timeout(
    client,
    monkeypatch,
):
    """Return HTTP 504 when Ollama takes too long to respond."""

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaTimeoutError("Ollama took too long to respond.")

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "The language model request timed out.",
    }


def test_assistant_chat_returns_502_for_invalid_ollama_response(
    client,
    monkeypatch,
):
    """Return HTTP 502 when Ollama returns an invalid response."""

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaInvalidResponseError("Ollama returned an empty answer.")

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model returned an invalid response.",
    }


def test_assistant_chat_returns_502_for_generic_ollama_error(
    client,
    monkeypatch,
):
    """Return HTTP 502 for unexpected Ollama request errors."""

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaRequestError("Ollama returned HTTP status 500: internal error")

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model service returned an unexpected error.",
    }


def test_assistant_chat_model_unavailable_uses_configured_model_name(
    client,
    monkeypatch,
):
    """Use the dynamically configured model name in the error response."""

    monkeypatch.setenv("OLLAMA_MODEL", "custom-model:7b")

    async def mock_ensure_model_available(self, model=None):
        raise OllamaModelUnavailableError(
            "Configured Ollama model is not installed: custom-model:7b"
        )

    monkeypatch.setattr(
        OllamaService,
        "ensure_model_available",
        mock_ensure_model_available,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Configured Ollama model is unavailable: custom-model:7b",
    }


def test_assistant_chat_logs_are_sanitized(
    client,
    monkeypatch,
    caplog,
):
    """Do not leak user messages or internal error details in logs or responses."""

    caplog.set_level(logging.WARNING)

    sensitive_message = "SENSITIVE_USER_MESSAGE_123"

    async def mock_ask(
        self,
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaConnectionError(
            "Could not connect to Ollama at http://localhost:11434."
        )

    monkeypatch.setattr(
        OllamaService,
        "ask",
        mock_ask,
    )

    response = client.post(
        "/assistant/chat",
        json={
            "message": sensitive_message,
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Ollama service is unavailable.",
    }

    response_text = response.text.lower()
    assert sensitive_message not in caplog.text
    assert "ollamaconnectionerror" not in response_text
    assert "localhost" not in response_text
    assert "traceback" not in response_text
    assert "could not connect" not in response_text
