"""Shared FastAPI test client fixture."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.rag import RetrievalResult


class EmptyRagService:
    """Keep the fast API test suite independent from local Ollama state."""

    async def retrieve(self, query: str, *, top_k: int | None = None):
        return RetrievalResult(query=query.strip(), results=[])


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        original_rag_service = test_client.app.state.rag_service
        test_client.app.state.rag_service = EmptyRagService()

        try:
            yield test_client
        finally:
            test_client.app.state.rag_service = original_rag_service
