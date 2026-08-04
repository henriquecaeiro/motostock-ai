"""End-to-end API workflow using a temporary SQLite database."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.database import initialize_database
from api.main import app
from api.repositories.sqlite_import import import_csv_files
from api.schemas.rag import RetrievalResult, RetrievedChunk
from src.data_preparation import prepare_processed_data


class FakeRagService:
    async def retrieve(self, query: str, *, top_k: int | None = None):
        return RetrievalResult(
            query=query,
            results=[
                RetrievedChunk(
                    chunk_id="e2e-architecture-001",
                    source="system_architecture.md",
                    title="System architecture",
                    section="Architecture",
                    content=(
                        "XGBoost predicts demand, the recommendation engine "
                        "calculates replenishment, and the LLM explains results."
                    ),
                    chunk_index=1,
                    score=0.91,
                )
            ],
        )


@pytest.mark.e2e
def test_complete_operational_and_assistant_workflow(tmp_path, monkeypatch):
    """Run initialization, data, ML, operations, RAG, tools and error paths."""

    database_path = tmp_path / "e2e.db"
    daily_path = tmp_path / "daily_product_sales.csv"
    modeling_path = tmp_path / "modeling_dataset.csv"
    raw_path = Path(__file__).parents[1] / "data" / "raw" / "motoretail.csv"

    prepare_processed_data(raw_path, daily_path, modeling_path)
    assert initialize_database(database_path) >= 1
    imported = import_csv_files(
        database_path,
        daily_sales_path=daily_path,
        modeling_data_path=modeling_path,
    )
    assert imported["products"] == 12
    assert imported["sales"] > 0
    assert imported["modeling_rows"] > 0

    monkeypatch.setenv("DATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "false")
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "rag"))

    with TestClient(app) as client:
        health = client.get("/health")
        products = client.get("/products")
        prediction = client.post(
            "/predict",
            json={"product_name": "Bag Delivery 45L", "horizon_days": 7},
        )
        recommendations = client.get("/recommendations", params={"horizon_days": 7})

        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert products.status_code == 200
        assert products.json()["count"] == 12
        assert prediction.status_code == 200
        assert len(prediction.json()["daily_forecast"]) == 7
        assert recommendations.status_code == 200
        assert recommendations.json()["count"] == 12

        sale = client.post(
            "/sales",
            json={
                "product_name": "Bag Delivery 45L",
                "sale_date": "2026-01-01",
                "quantity_sold": 2,
                "unit_price_brl": 182.5,
                "external_id": "e2e-sale-1",
            },
        )
        refresh = client.post(
            "/recommendations/refresh",
            json={"horizon_days": 7},
        )
        latest = client.get("/recommendations/latest", params={"horizon_days": 7})

        assert sale.status_code == 201
        assert sale.json()["inserted"] == 1
        assert refresh.status_code == 200
        assert refresh.json()["status"] == "completed"
        assert refresh.json()["recommendations_count"] == 12
        assert latest.status_code == 200
        assert latest.json()["count"] == 12

        monkeypatch.setattr(client.app.state, "rag_service", FakeRagService())

        async def fake_ask(prompt, *, system_prompt=None, model=None, options=None):
            assert "system_architecture.md" in prompt
            return "The documented architecture separates forecasting, recommendations and explanation."

        monkeypatch.setattr(client.app.state.ollama_service, "ask", fake_ask)
        rag_response = client.post(
            "/assistant/chat",
            json={"message": "How does the architecture work?"},
        )
        tool_response = client.post(
            "/assistant/chat",
            json={"message": "List products"},
        )

        assert rag_response.status_code == 200
        assert rag_response.json()["sources"][0]["source"] == "system_architecture.md"
        assert rag_response.json()["tools_used"] == []
        assert tool_response.status_code == 200
        assert tool_response.json()["tools_used"] == []
        assert "Encontrei 12 produtos" in tool_response.json()["answer"]

        unknown_product = client.post(
            "/predict",
            json={"product_name": "Unknown product", "horizon_days": 7},
        )
        invalid_horizon = client.post(
            "/predict",
            json={"product_name": "Bag Delivery 45L", "horizon_days": 31},
        )
        invalid_sale = client.post(
            "/sales",
            json={
                "product_name": "Bag Delivery 45L",
                "sale_date": "2026-01-02",
                "quantity_sold": 1,
                "unit_price_brl": 180,
            },
        )

        assert unknown_product.status_code == 404
        assert invalid_horizon.status_code == 422
        assert invalid_sale.status_code == 422
