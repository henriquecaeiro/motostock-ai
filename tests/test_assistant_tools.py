"""HTTP-level tests for deterministic assistant tool orchestration."""

from __future__ import annotations

from api.schemas.rag import RetrievalResult
from api.services.tool_service import ToolService


class EmptyOrFakeRag:
    def __init__(self, result=None):
        self.result = result

    async def retrieve(self, query: str, *, top_k=None):
        return self.result or RetrievalResult(query=query, results=[])


class ControlledRepository:
    products = ["Bag Delivery 45L", "Capacete LS2", "Suporte Celular Moto"]

    def list_products(self):
        return list(self.products)

    def product_exists(self, product_name):
        return product_name in self.products


class ControlledForecastService:
    def predict_product(self, *, product_name, horizon_days):
        return {
            "product_name": product_name,
            "horizon_days": horizon_days,
            "forecasted_demand_units": 19,
        }

    def get_forecast_summary(self, *, horizon_days, limit):
        return {
            "selected_model": "fake-model",
            "horizon_days": horizon_days,
            "total_products": 3,
            "total_forecasted_demand_units": 42,
            "top_products": [
                {
                    "product_name": "Bag Delivery 45L",
                    "forecasted_demand_units": 18,
                    "forecasted_demand_raw": 17.6,
                    "forecasted_demand_non_negative": 18.0,
                },
                {
                    "product_name": "Capacete LS2",
                    "forecasted_demand_units": 14,
                    "forecasted_demand_raw": 13.7,
                    "forecasted_demand_non_negative": 14.0,
                },
            ][:limit],
        }


class ControlledRecommendationService:
    def get_recommendations(
        self,
        *,
        horizon_days,
        stock_status=None,
        product_name=None,
    ):
        recommendations = [
            {
                "product_name": "Bag Delivery 45L",
                "forecast_horizon_days": horizon_days,
                "forecasted_demand_units": 11,
                "current_stock": 3,
                "safety_stock": 4,
                "required_stock": 8,
                "recommended_purchase_quantity": 5,
                "supplier_lead_time_days": 2,
                "stock_status": "critical",
                "priority_score": 97,
            }
        ]

        if product_name:
            recommendations = [
                item
                for item in recommendations
                if item["product_name"] == product_name
            ]

        if stock_status:
            recommendations = [
                item
                for item in recommendations
                if item["stock_status"] == stock_status
            ]

        return {
            "horizon_days": horizon_days,
            "count": len(recommendations),
            "recommendations": recommendations,
        }


class RecordingRag:
    def __init__(self, result=None):
        self.called = False
        self.result = result

    async def retrieve(self, query: str, *, top_k=None):
        self.called = True
        return self.result or RetrievalResult(query=query, results=[])


def make_controlled_tool_service() -> ToolService:
    return ToolService(
        repository=ControlledRepository(),
        forecast_service=ControlledForecastService(),
        recommendation_service=ControlledRecommendationService(),
    )


def test_chat_list_products_uses_read_only_tool(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "List products"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "Encontrei 12 produtos" in payload["answer"]
    assert "Bag Delivery 45L" in payload["answer"]
    assert payload["sources"] == []


def test_chat_forecast_tool_returns_service_value_for_known_product(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Bag Delivery 45L for 7 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "Bag Delivery 45L" in payload["answer"]
    assert "7 dias" in payload["answer"]
    assert "unidades" in payload["answer"]


def test_chat_unknown_product_does_not_invent_forecast(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Unknown Product for 14 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "inválid" in payload["answer"].lower()
    assert "Unknown Product" not in payload["answer"]


def test_chat_invalid_horizon_is_rejected_without_tool_execution(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Forecast Bag Delivery 45L for 45 days"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "inválid" in payload["answer"].lower()


def test_chat_recommendation_status_filter_uses_exact_service_output(client):
    response = client.post(
        "/assistant/chat",
        json={"message": "Show current critical recommendations"},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert "Não há produtos críticos" in payload["answer"]
    assert "get_recommendations" not in payload["answer"]


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
    assert payload["tools_used"] == []
    assert "Unidades totais para compra recomendada: 2" in payload["answer"]
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
    assert payload["tools_used"] == []
    assert payload["sources"][0]["source"] == "stock_recommendation_rules.md"
    assert "get_recommendations" not in payload["answer"]


def test_chat_portuguese_top_five_executes_current_data_internally(client, monkeypatch):
    ollama = client.app.state.ollama_service
    ask_called = False

    async def mock_ask(prompt, *, system_prompt=None, model=None, options=None):
        nonlocal ask_called
        ask_called = True
        return "não deveria ser chamado"

    monkeypatch.setattr(ollama, "ask", mock_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "Me mande a lista dos 5 produtos mais críticos."},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["tools_used"] == []
    assert ask_called is False
    assert "get_recommendations" not in payload["answer"]
    assert "produtos críticos" in payload["answer"]


def test_chat_purchase_query_returns_exact_recommendation_values(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        client.app.state,
        "tool_service",
        make_controlled_tool_service(),
    )
    rag = RecordingRag()
    monkeypatch.setattr(client.app.state, "rag_service", rag)

    async def fail_ask(prompt, *, system_prompt=None, model=None, options=None):
        raise AssertionError("Ollama.ask must not be called for a direct tool query")

    monkeypatch.setattr(client.app.state.ollama_service, "ask", fail_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "Quanto devo comprar de Bag Delivery 45L?"},
    )

    answer = response.json()["answer"]

    assert response.status_code == 200
    assert "Bag Delivery 45L" in answer
    assert "5 unidades" in answer
    assert "3 unidades" in answer
    assert "11 unidades" in answer
    assert "4 unidades" in answer
    assert "8 unidades" in answer
    assert "2 dias" in answer
    assert "97" in answer
    assert "indisponíveis" not in answer
    assert response.json()["tools_used"] == []
    assert response.json()["sources"] == []
    assert rag.called is False


def test_chat_general_forecast_returns_controlled_summary_without_llm(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        client.app.state,
        "tool_service",
        make_controlled_tool_service(),
    )
    rag = RecordingRag()
    monkeypatch.setattr(client.app.state, "rag_service", rag)

    async def fail_ask(prompt, *, system_prompt=None, model=None, options=None):
        raise AssertionError("Ollama.ask must not be called for a direct tool query")

    monkeypatch.setattr(client.app.state.ollama_service, "ask", fail_ask)

    response = client.post(
        "/assistant/chat",
        json={"message": "Qual é a previsão para os próximos 14 dias?"},
    )

    answer = response.json()["answer"]

    assert response.status_code == 200
    assert "42 unidades" in answer
    assert "14 dias" in answer
    assert "Bag Delivery 45L" in answer
    assert "Capacete LS2" in answer
    assert "product_name" not in answer
    assert response.json()["tools_used"] == []
    assert rag.called is False


def test_chat_specific_forecast_keeps_product_and_horizon(client, monkeypatch):
    monkeypatch.setattr(
        client.app.state,
        "tool_service",
        make_controlled_tool_service(),
    )

    async def fail_ask(prompt, *, system_prompt=None, model=None, options=None):
        raise AssertionError("Ollama.ask must not be called for a direct tool query")

    monkeypatch.setattr(client.app.state.ollama_service, "ask", fail_ask)

    response = client.post(
        "/assistant/chat",
        json={
            "message": "Qual é a previsão de Bag Delivery 45L para os próximos 14 dias?"
        },
    )

    answer = response.json()["answer"]

    assert response.status_code == 200
    assert "Bag Delivery 45L" in answer
    assert "14 dias" in answer
    assert "19 unidades" in answer


def test_chat_mixed_purchase_query_keeps_current_values_and_rag_sources(
    client,
    monkeypatch,
):
    from api.schemas.rag import RetrievedChunk

    result = RetrievalResult(
        query="Por que devo comprar a quantidade recomendada de Bag Delivery 45L?",
        results=[
            RetrievedChunk(
                chunk_id="purchase-rules-001",
                source="stock_recommendation_rules.md",
                title="Stock Rules",
                section="Purchase Quantity",
                content="The recommendation engine uses current stock and forecast demand.",
                chunk_index=1,
                score=0.88,
            )
        ],
    )
    monkeypatch.setattr(
        client.app.state,
        "tool_service",
        make_controlled_tool_service(),
    )
    rag = RecordingRag(result)
    monkeypatch.setattr(client.app.state, "rag_service", rag)

    async def fail_ask(prompt, *, system_prompt=None, model=None, options=None):
        raise AssertionError("Ollama.ask must not modify verified tool values")

    monkeypatch.setattr(client.app.state.ollama_service, "ask", fail_ask)

    response = client.post(
        "/assistant/chat",
        json={
            "message": "Por que devo comprar a quantidade recomendada de Bag Delivery 45L?"
        },
    )

    answer = response.json()["answer"]

    assert response.status_code == 200
    assert "Bag Delivery 45L" in answer
    assert "5 unidades" in answer
    assert "3 unidades" in answer
    assert response.json()["sources"][0]["source"] == "stock_recommendation_rules.md"
    assert rag.called is True
    assert "get_recommendations" not in answer
