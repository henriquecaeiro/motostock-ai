"""Assistant endpoint tests."""

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaService,
    OllamaTimeoutError,
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
        "detail": (
            "The AI assistant is temporarily unavailable "
            "because Ollama could not be reached."
        )
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
    assert response.json() == {"detail": "The AI assistant took too long to respond."}
