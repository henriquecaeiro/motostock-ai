"""Recommendation endpoint tests."""


def test_recommendations_returns_200(client):
    response = client.get("/recommendations")
    assert response.status_code == 200


def test_recommendation_product_names_are_unique(client):
    recommendations = client.get("/recommendations").json()["recommendations"]
    product_names = [item["product_name"] for item in recommendations]
    assert len(product_names) == len(set(product_names))


def test_recommendation_quantities_are_non_negative(client):
    recommendations = client.get("/recommendations").json()["recommendations"]
    for item in recommendations:
        assert item["forecasted_demand_units"] >= 0
        assert item["current_stock"] >= 0
        assert item["safety_stock"] >= 0
        assert item["required_stock"] >= 0
        assert item["recommended_purchase_quantity"] >= 0


def test_recommended_purchase_quantities_are_integers(client):
    recommendations = client.get("/recommendations").json()["recommendations"]
    for item in recommendations:
        assert isinstance(item["recommended_purchase_quantity"], int)


def test_filtering_by_status_works(client):
    all_recommendations = client.get("/recommendations").json()["recommendations"]
    if not all_recommendations:
        return

    status = all_recommendations[0]["stock_status"]
    filtered = client.get("/recommendations", params={"stock_status": status}).json()
    assert filtered["count"] >= 1
    assert all(item["stock_status"] == status for item in filtered["recommendations"])


def test_invalid_stock_status_returns_422(client):
    response = client.get("/recommendations", params={"stock_status": "invalid"})
    assert response.status_code == 422


def test_unknown_product_filter_returns_404(client):
    response = client.get(
        "/recommendations",
        params={"product_name": "Unknown Product XYZ"},
    )
    assert response.status_code == 404


def test_summary_counts_are_consistent_with_recommendations(client):
    recommendations_payload = client.get("/recommendations").json()
    summary_payload = client.get("/recommendations/summary").json()

    recommendations = recommendations_payload["recommendations"]
    status_counts = {
        "critical": 0,
        "warning": 0,
        "healthy": 0,
        "overstock": 0,
    }

    for item in recommendations:
        status_counts[item["stock_status"]] += 1

    assert summary_payload["total_products"] == recommendations_payload["count"]
    assert summary_payload["critical_products"] == status_counts["critical"]
    assert summary_payload["warning_products"] == status_counts["warning"]
    assert summary_payload["healthy_products"] == status_counts["healthy"]
    assert summary_payload["overstock_products"] == status_counts["overstock"]
    assert summary_payload["total_recommended_purchase_units"] == sum(
        item["recommended_purchase_quantity"] for item in recommendations
    )
