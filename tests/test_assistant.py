"""Assistant endpoint tests."""

from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaService,
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
