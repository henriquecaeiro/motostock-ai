"""HTTP-level tests for deterministic assistant tool orchestration."""

from __future__ import annotations

from api.schemas.rag import RetrievalResult


class EmptyOrFakeRag:
    def __init__(self, result=None):
        self.result = result

    async def retrieve(self, query: str, *, top_k=None):
        return self.result or RetrievalResult(query=query, results=[])


def test_chat_list_products_uses_read_only_tool(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "List products"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == ["list_products"]
    assert '"count": 12' in payload["answer"]
    assert "Bag Delivery 45L" in payload["answer"]
    assert payload["sources"] == []


def test_chat_forecast_tool_returns_service_value_for_known_product(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Bag Delivery 45L for 7 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == ["forecast_product"]
    assert '"product_name": "Bag Delivery 45L"' in payload["answer"]
    assert '"horizon_days": 7' in payload["answer"]


def test_chat_unknown_product_does_not_invent_forecast(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Unknown Product for 14 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "invalid" in payload["answer"].lower()
    assert "Unknown Product" not in payload["answer"]


def test_chat_invalid_horizon_is_rejected_without_tool_execution(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Bag Delivery 45L for 45 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "invalid" in payload["answer"].lower()


def test_chat_recommendation_status_filter_uses_exact_service_output(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Show current critical recommendations"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == ["get_recommendations"]
    assert '"stock_status": "critical"' in payload["answer"]
    assert '"recommendations": []' in payload["answer"]


def test_chat_summary_uses_tool_without_rag_or_llm(client, monkeypatch):
    ollama = client.app.state.ollama_service
    ask_called = False

    async def mock_ask(prompt, *, system_prompt=None, model=None, options=None):
        nonlocal ask_called
        ask_called = True
        return "should not be called"

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "Show recommendation summary"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == ["get_recommendation_summary"]
    assert '"total_recommended_purchase_units": 2' in payload["answer"]
    assert payload["sources"] == []
    assert ask_called is False


def test_chat_conceptual_question_does_not_call_a_tool(client, monkeypatch):
    ollama = client.app.state.ollama_service
    ask_called = False

    async def mock_ask(prompt, *, system_prompt=None, model=None, options=None):
        nonlocal ask_called
        ask_called = True
        return "Safety stock is a documented buffer."

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "How does the recommendation formula work?"},
    )

    assert response.status_code == 200
    assert response.json()["tools_used"] == []
    assert ask_called is True


def test_chat_mixed_question_returns_tools_and_rag_sources(client, monkeypatch):
    from api.schemas.rag import RetrievedChunk

    result = RetrievalResult(
        query="Why is the current recommendation for Bag Delivery 45L important?",
        results=[
            RetrievedChunk(
                chunk_id="rules-001",
                source="stock_recommendation_rules.md",
                title="Stock Rules",
                section="Critical",
                content="Critical status means current stock is below forecasted demand.",
                chunk_index=1,
                score=0.8,
            )
        ],
    )
    monkeypatch.setattr(client.app.state, "rag_service", EmptyOrFakeRag(result))

    response = client.post(
        "/assistant/chat",
        json={
            "message": "Why is the current recommendation for Bag Delivery 45L important?"
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == ["get_recommendations"]
    assert payload["sources"][0]["source"] == "stock_recommendation_rules.md"
    assert '"recommendations"' in payload["answer"]
