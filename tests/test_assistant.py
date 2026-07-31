"""Assistant endpoint tests."""

import logging
from pathlib import Path

from api.prompts import load_system_prompt
from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaTimeoutError,
)


def _ollama_service(client):
    """Return the shared OllamaService instance from app state."""

    return client.app.state.ollama_service


def test_assistant_health_returns_ok(client, monkeypatch):
    """Return HTTP 200 when Ollama and the configured model are available."""

    ollama = _ollama_service(client)
    monkeypatch.setattr(ollama, "default_model", "qwen3:4b")

    async def mock_list_models():
        return ["qwen3:4b"]

    monkeypatch.setattr(ollama, "list_models", mock_list_models)

    response = client.get("/assistant/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ollama_available"] is True
    assert payload["model_available"] is True
    assert payload["model"] == "qwen3:4b"


def test_assistant_health_uses_configured_model_from_shared_service(
    client,
    monkeypatch,
):
    """Expose the model configured on the shared OllamaService instance."""

    ollama = _ollama_service(client)
    configured_model = "qwen3:test-model"
    monkeypatch.setattr(ollama, "default_model", configured_model)

    async def mock_list_models():
        return [configured_model]

    monkeypatch.setattr(ollama, "list_models", mock_list_models)

    response = client.get("/assistant/health")

    assert response.status_code == 200
    assert response.json()["model"] == configured_model


def test_assistant_health_returns_503_when_model_is_missing(client, monkeypatch):
    """Return HTTP 503 when Ollama is available but the model is missing."""

    ollama = _ollama_service(client)
    monkeypatch.setattr(ollama, "default_model", "qwen3:4b")

    async def mock_list_models():
        return ["qwen3:8b"]

    monkeypatch.setattr(ollama, "list_models", mock_list_models)

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

    ollama = _ollama_service(client)
    monkeypatch.setattr(ollama, "default_model", "qwen3:4b")

    async def mock_list_models():
        raise OllamaConnectionError("Could not connect to Ollama.")

    monkeypatch.setattr(ollama, "list_models", mock_list_models)

    response = client.get("/assistant/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "ollama_unavailable",
        "ollama_available": False,
        "model": "qwen3:4b",
        "model_available": False,
    }


def test_assistant_chat_returns_answer(client, monkeypatch):
    """Return the assistant answer using the configured model."""

    ollama = _ollama_service(client)
    monkeypatch.setattr(ollama, "default_model", "qwen3:4b")

    async def mock_ask(
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

    monkeypatch.setattr(ollama, "ask", mock_ask)

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
    payload = response.json()
    assert isinstance(payload["answer"], str)
    assert payload["answer"]
    assert payload["model"] == "qwen3:4b"
    assert payload["tools_used"] == []
    assert payload["sources"] == []


def test_assistant_chat_tools_used_starts_empty(client, monkeypatch):
    """Confirm tools_used remains an empty list in successful responses."""

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        return "MotoStock AI helps with demand forecasting."

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "What is MotoStock AI?"},
    )

    assert response.status_code == 200
    assert response.json()["tools_used"] == []


def test_assistant_chat_sources_starts_empty(client, monkeypatch):
    """Confirm sources remains an empty list in successful responses."""

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        return "MotoStock AI helps with demand forecasting."

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "What is MotoStock AI?"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_assistant_chat_system_prompt_is_passed_but_not_exposed(
    client,
    monkeypatch,
    caplog,
):
    """Pass the system prompt to the service without exposing it in logs or HTTP."""

    caplog.set_level(logging.DEBUG)
    ollama = _ollama_service(client)
    expected_system_prompt = load_system_prompt()

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        assert system_prompt == expected_system_prompt
        return "Conceptual answer."

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "What is safety stock?"},
    )

    assert response.status_code == 200
    assert expected_system_prompt not in response.text
    assert expected_system_prompt not in caplog.text


def test_assistant_chat_removes_external_whitespace(client, monkeypatch):
    """Remove unnecessary whitespace before sending the prompt."""

    ollama = _ollama_service(client)
    received_prompt = None

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        nonlocal received_prompt
        received_prompt = prompt
        return "Safety stock is extra inventory."

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={
            "message": "   What is safety stock?   ",
        },
    )

    assert response.status_code == 200
    assert received_prompt == "What is safety stock?"


def test_assistant_chat_rejects_empty_string_message(client):
    """Reject an empty message string."""

    response = client.post(
        "/assistant/chat",
        json={
            "message": "",
        },
    )

    assert response.status_code == 422


def test_assistant_chat_rejects_whitespace_only_message(client):
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

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaConnectionError("Could not connect to Ollama.")

    monkeypatch.setattr(ollama, "ask", mock_ask)

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

    ollama = _ollama_service(client)
    configured_model = "qwen3:missing"
    monkeypatch.setattr(ollama, "default_model", configured_model)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaModelUnavailableError(configured_model)

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": f"Configured Ollama model is unavailable: {configured_model}",
    }


def test_assistant_chat_returns_504_on_timeout(client, monkeypatch):
    """Return HTTP 504 when Ollama takes too long to respond."""

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaTimeoutError("Ollama took too long to respond.")

    monkeypatch.setattr(ollama, "ask", mock_ask)

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

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaInvalidResponseError("Ollama returned an empty answer.")

    monkeypatch.setattr(ollama, "ask", mock_ask)

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

    ollama = _ollama_service(client)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaRequestError("Ollama returned HTTP status 500: internal error")

    monkeypatch.setattr(ollama, "ask", mock_ask)

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

    ollama = _ollama_service(client)
    configured_model = "custom-model:7b"
    monkeypatch.setattr(ollama, "default_model", configured_model)

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaModelUnavailableError(configured_model)

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={
            "message": "What is MotoStock AI?",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": f"Configured Ollama model is unavailable: {configured_model}",
    }


def test_assistant_endpoints_use_shared_ollama_service(client, monkeypatch):
    """Use the same shared OllamaService instance from app.state."""

    ollama = _ollama_service(client)

    async def mock_list_models():
        return [ollama.default_model]

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        return "Shared service answer."

    monkeypatch.setattr(ollama, "list_models", mock_list_models)
    monkeypatch.setattr(ollama, "ask", mock_ask)

    health_response = client.get("/assistant/health")
    chat_response = client.post(
        "/assistant/chat",
        json={"message": "What is MotoStock AI?"},
    )

    assert health_response.status_code == 200
    assert chat_response.status_code == 200
    assert chat_response.json()["answer"] == "Shared service answer."


def test_assistant_routes_do_not_instantiate_ollama_service():
    """Keep OllamaService creation in the application lifespan only."""

    source = Path("api/routes/assistant.py").read_text(encoding="utf-8")

    assert "OllamaService()" not in source
    assert "async with OllamaService()" not in source
    assert "await ollama.close()" not in source
    assert "request.app.state.ollama_service" in source


def test_assistant_chat_logs_are_sanitized(client, monkeypatch, caplog):
    """Do not leak user messages or internal error details in logs or responses."""

    caplog.set_level(logging.WARNING)
    ollama = _ollama_service(client)
    sensitive_message = "SENSITIVE_USER_MESSAGE_123"

    async def mock_ask(
        prompt,
        *,
        system_prompt=None,
        model=None,
        options=None,
    ):
        raise OllamaConnectionError(
            "Could not connect to Ollama at http://localhost:11434."
        )

    monkeypatch.setattr(ollama, "ask", mock_ask)

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
