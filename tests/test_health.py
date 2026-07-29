"""Health endpoint tests."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_ok(client):
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_model_loaded_is_true(client):
    response = client.get("/health")
    assert response.json()["model_loaded"] is True
