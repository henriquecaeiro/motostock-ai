"""MotoBoy POS inventory snapshot contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from api.exceptions import ServiceUnavailableError
from api.main import app
from api.repositories.sqlite_repository import SqliteRepository


pytestmark = pytest.mark.integration


def _snapshot(
    *,
    key: str,
    quantity: int,
    observed_at: str = "2026-08-03T18:30:00Z",
    product_name: str = "Bag Delivery 45L",
) -> dict:
    return {
        "product_name": product_name,
        "quantity_on_hand": quantity,
        "supplier_lead_time_days": 7,
        "observed_at": observed_at,
        "external_id": f"motoboy-pos:{key}",
        "idempotency_key": f"motoboy-pos:{key}",
    }


@pytest.fixture
def sqlite_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "operations.db"))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "true")

    with TestClient(app) as client:
        yield client


def test_valid_snapshot_is_persisted_and_retry_is_idempotent(sqlite_client) -> None:
    payload = _snapshot(key="stock-movement:27", quantity=10)

    first = sqlite_client.post("/inventory/snapshots", json=payload)
    retry = sqlite_client.post("/inventory/snapshots", json=payload)

    assert first.status_code == 201
    assert first.json()["inserted"] == 1
    assert first.json()["updated"] == 1
    assert retry.status_code == 201
    assert retry.json()["skipped"] == 1
    assert retry.json()["inserted"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"product_name": "Unknown Product"},
        {"quantity_on_hand": -1},
        {"supplier_lead_time_days": 0},
        {"observed_at": "2026-08-03T18:30:00"},
        {"external_id": None, "idempotency_key": None},
    ],
)
def test_snapshot_validation_and_catalog_contract(sqlite_client, changes) -> None:
    payload = _snapshot(key="invalid-case", quantity=10)
    payload.update(changes)

    response = sqlite_client.post("/inventory/snapshots", json=payload)

    assert response.status_code in {404, 422}
    assert "Traceback" not in response.text


def test_batch_is_transactional_and_duplicate_events_are_skipped(sqlite_client) -> None:
    payload = {
        "snapshots": [
            _snapshot(key="batch-1", quantity=8),
            _snapshot(
                key="batch-2",
                quantity=6,
                product_name="Capacete Pro Tork",
            ),
        ]
    }

    response = sqlite_client.post("/inventory/snapshots/batch", json=payload)
    retry = sqlite_client.post("/inventory/snapshots/batch", json=payload)

    assert response.status_code == 201
    assert response.json()["inserted"] == 2
    assert retry.status_code == 201
    assert retry.json()["skipped"] == 2


def test_older_event_never_replaces_newer_snapshot(sqlite_client) -> None:
    newer = _snapshot(
        key="newer",
        quantity=12,
        observed_at="2026-08-03T19:00:00Z",
    )
    older = _snapshot(
        key="older",
        quantity=3,
        observed_at="2026-08-03T18:00:00Z",
    )

    assert sqlite_client.post("/inventory/snapshots", json=newer).status_code == 201
    older_response = sqlite_client.post("/inventory/snapshots", json=older)

    assert older_response.status_code == 201
    assert older_response.json()["inserted"] == 1
    assert older_response.json()["updated"] == 0

    state = sqlite_client.app.state.repository.get_latest_business_state()
    bag = state[state["product_name"] == "Bag Delivery 45L"].iloc[0]
    assert bag["current_stock"] == 12


def test_snapshot_is_used_by_recommendation_engine(sqlite_client) -> None:
    response = sqlite_client.post(
        "/inventory/snapshots",
        json=_snapshot(
            key="recommendation-input",
            quantity=99,
            observed_at="2026-08-03T20:00:00Z",
        ),
    )
    assert response.status_code == 201

    recommendations = sqlite_client.get("/recommendations").json()["recommendations"]
    bag = next(
        item for item in recommendations if item["product_name"] == "Bag Delivery 45L"
    )
    assert bag["current_stock"] == 99


def test_snapshot_persists_after_repository_reopen(sqlite_client) -> None:
    response = sqlite_client.post(
        "/inventory/snapshots",
        json=_snapshot(key="persisted", quantity=17),
    )
    assert response.status_code == 201

    reopened = SqliteRepository(sqlite_client.app.state.repository.database_path)
    state = reopened.get_latest_business_state().set_index("product_name")
    assert state.loc["Bag Delivery 45L", "current_stock"] == 17


def test_database_failure_returns_safe_service_error(sqlite_client, monkeypatch) -> None:
    def fail(_records):
        raise ServiceUnavailableError("Inventory snapshot ingestion is unavailable.")

    monkeypatch.setattr(
        sqlite_client.app.state.repository,
        "insert_inventory_snapshots",
        fail,
    )
    response = sqlite_client.post(
        "/inventory/snapshots",
        json=_snapshot(key="database-failure", quantity=10),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Inventory snapshot ingestion is unavailable."
    assert "Traceback" not in response.text
