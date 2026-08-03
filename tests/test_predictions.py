"""Prediction endpoint tests."""

import pytest


@pytest.fixture(scope="module")
def known_product(client):
    return client.get("/products").json()["products"][0]


def test_predict_known_product_returns_200(client, known_product):
    response = client.post(
        "/predict",
        json={"product_name": known_product},
    )
    assert response.status_code == 200


def test_predict_default_horizon_returns_14_days(client, known_product):
    payload = client.post(
        "/predict",
        json={"product_name": known_product},
    ).json()
    assert payload["horizon_days"] == 14
    assert len(payload["daily_forecast"]) == 14


def test_predict_custom_horizon_works(client, known_product):
    payload = client.post(
        "/predict",
        json={"product_name": known_product, "horizon_days": 7},
    ).json()
    assert payload["horizon_days"] == 7
    assert len(payload["daily_forecast"]) == 7


def test_predicted_demand_units_is_integer(client, known_product):
    payload = client.post(
        "/predict",
        json={"product_name": known_product},
    ).json()
    assert isinstance(payload["forecasted_demand_units"], int)


def test_daily_non_negative_predictions_are_never_below_zero(client, known_product):
    payload = client.post(
        "/predict",
        json={"product_name": known_product},
    ).json()
    for day in payload["daily_forecast"]:
        assert day["forecast_non_negative"] >= 0


def test_unknown_product_returns_404(client):
    response = client.post(
        "/predict",
        json={"product_name": "Unknown Product XYZ"},
    )
    assert response.status_code == 404


def test_horizon_zero_returns_422(client, known_product):
    response = client.post(
        "/predict",
        json={"product_name": known_product, "horizon_days": 0},
    )
    assert response.status_code == 422


def test_horizon_above_30_returns_422(client, known_product):
    response = client.post(
        "/predict",
        json={"product_name": known_product, "horizon_days": 31},
    )
    assert response.status_code == 422
