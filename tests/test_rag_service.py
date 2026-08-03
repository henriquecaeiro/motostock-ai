"""Tests for query-to-retrieval orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from api.config import Settings
from api.schemas.rag import KnowledgeChunk
from api.services.rag_service import RagService
from api.services.vector_store_service import VectorStoreService, VectorStoreNotFoundError


class FakeEmbeddingService:
    def __init__(self, vector=None):
        self.vector = vector or [1.0, 0.0]
        self.queries: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector


def make_store(tmp_path: Path, *, max_top_k: int = 5) -> VectorStoreService:
    settings = Settings(
        rag_storage_path=tmp_path / "rag",
        rag_default_top_k=2,
        rag_max_top_k=max_top_k,
    )
    chunks = [
        KnowledgeChunk(
            chunk_id="one-001",
            source="one.md",
            title="One",
            section="Overview",
            content="First content",
            chunk_index=1,
        ),
        KnowledgeChunk(
            chunk_id="two-001",
            source="two.md",
            title="Two",
            section="Overview",
            content="Second content",
            chunk_index=1,
        ),
    ]
    store = VectorStoreService(settings=settings)
    store.build_index(
        chunks,
        [[1.0, 0.0], [0.0, 1.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )
    return store


def test_retrieve_embeds_trimmed_query_and_returns_structured_results(tmp_path: Path) -> None:
    embeddings = FakeEmbeddingService()
    store = make_store(tmp_path)
    service = RagService(
        embeddings,
        store,
        settings=Settings(
            rag_storage_path=tmp_path / "rag",
            rag_default_top_k=2,
            rag_max_top_k=5,
        ),
    )

    result = asyncio.run(service.retrieve("  demand question  ", top_k=1))

    assert embeddings.queries == ["demand question"]
    assert result.query == "demand question"
    assert len(result.results) == 1
    assert result.results[0].chunk_id == "one-001"
    assert result.results[0].score == pytest.approx(1.0)
    assert result.index_version == 1


def test_retrieve_applies_optional_score_threshold(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    service = RagService(
        FakeEmbeddingService([1.0, 1.0]),
        store,
        settings=Settings(
            rag_storage_path=tmp_path / "rag",
            rag_default_top_k=2,
            rag_max_top_k=5,
            rag_min_score=0.8,
        ),
    )

    result = asyncio.run(service.retrieve("question"))

    assert len(result.results) == 0


@pytest.mark.parametrize("query", ["", "   ", None])
def test_retrieve_rejects_empty_or_non_string_queries(tmp_path: Path, query) -> None:
    service = RagService(FakeEmbeddingService(), make_store(tmp_path))

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(service.retrieve(query))


def test_retrieve_propagates_missing_index(tmp_path: Path) -> None:
    settings = Settings(
        rag_storage_path=tmp_path / "rag",
        rag_default_top_k=2,
        rag_max_top_k=5,
    )
    store = VectorStoreService(settings=settings)
    service = RagService(FakeEmbeddingService(), store, settings=settings)

    with pytest.raises(VectorStoreNotFoundError):
        asyncio.run(service.retrieve("question"))
