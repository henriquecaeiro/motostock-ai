"""Products endpoint tests."""


def test_products_returns_200(client):
    response = client.get("/products")
    assert response.status_code == 200


def test_products_list_is_not_empty(client):
    payload = client.get("/products").json()
    assert payload["count"] > 0
    assert len(payload["products"]) > 0


def test_products_are_unique_and_sorted(client):
    products = client.get("/products").json()["products"]
    assert products == sorted(products)
    assert len(products) == len(set(products))
