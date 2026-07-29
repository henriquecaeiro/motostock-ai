"""Product listing endpoint."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["Products"])


@router.get(
    "/products",
    summary="List known products",
    description="Return all unique product names from the current sales dataset.",
)
def list_products(request: Request) -> dict:
    repository = request.app.state.repository
    products = repository.list_products()
    return {
        "count": len(products),
        "products": products,
    }
