"""SQLite-backed sales and refresh API integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from api.main import app


pytestmark = pytest.mark.integration


def test_sales_endpoint_validates_keys_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "operations.db"))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "true")

    with TestClient(app) as client:
        sale = {
            "product_name": "Bag Delivery 45L",
            "sale_date": "2026-06-01",
            "quantity_sold": 2,
            "unit_price_brl": 182.5,
            "external_id": "integration-sale-1",
        }

        first = client.post("/sales", json=sale)
        duplicate = client.post("/sales", json=sale)
        missing_key = client.post(
            "/sales",
            json={**sale, "external_id": None},
        )

        assert first.status_code == 201
        assert first.json()["inserted"] == 1
        assert duplicate.status_code == 201
        assert duplicate.json()["skipped"] == 1
        assert missing_key.status_code == 422


def test_sales_batch_uses_header_key_and_rolls_back_on_unknown_product(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "operations.db"))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "true")

    with TestClient(app) as client:
        batch = {
            "sales": [
                {
                    "product_name": "Bag Delivery 45L",
                    "sale_date": "2026-06-02",
                    "quantity_sold": 1,
                    "unit_price_brl": 180,
                },
                {
                    "product_name": "Bag Delivery 80L",
                    "sale_date": "2026-06-02",
                    "quantity_sold": 1,
                    "unit_price_brl": 270,
                },
            ]
        }
        response = client.post(
            "/sales/batch",
            json=batch,
            headers={"Idempotency-Key": "integration-batch"},
        )
        duplicate = client.post(
            "/sales/batch",
            json=batch,
            headers={"Idempotency-Key": "integration-batch"},
        )

        assert response.status_code == 201
        assert response.json()["inserted"] == 2
        assert duplicate.json()["skipped"] == 2

        before = len(client.app.state.repository.load_daily_sales())
        failed = client.post(
            "/sales/batch",
            json={
                "sales": [
                    {
                        "product_name": "Bag Delivery 45L",
                        "sale_date": "2026-06-03",
                        "quantity_sold": 1,
                        "unit_price_brl": 180,
                        "external_id": "rollback-valid",
                    },
                    {
                        "product_name": "Does Not Exist",
                        "sale_date": "2026-06-03",
                        "quantity_sold": 1,
                        "unit_price_brl": 180,
                        "external_id": "rollback-invalid",
                    },
                ]
            },
        )

        assert failed.status_code == 404
        assert len(client.app.state.repository.load_daily_sales()) == before


def test_refresh_rebuilds_features_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "operations.db"))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "true")

    with TestClient(app) as client:
        response = client.post("/recommendations/refresh", json={"horizon_days": 7})
        repeated = client.post("/recommendations/refresh", json={"horizon_days": 7})

        assert response.status_code == 200
        first = response.json()
        assert first["status"] == "completed"
        assert first["features_rows"] > 0
        assert first["recommendations_count"] == 12
        assert repeated.json()["run_key"] == first["run_key"]
        assert repeated.json()["features_rows"] == first["features_rows"]

        latest = client.get("/recommendations/latest", params={"horizon_days": 7})
        stored_run = client.app.state.repository.get_application_run(first["run_key"])
        assert latest.status_code == 200
        assert latest.json()["count"] == 12
        assert stored_run["status"] == "completed"
