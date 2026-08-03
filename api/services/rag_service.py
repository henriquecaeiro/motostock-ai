"""Retrieval orchestration for the local MotoStock AI knowledge base."""

from __future__ import annotations

from api.config import Settings, load_settings
from api.schemas.rag import RetrievalResult
from api.services.embedding_service import EmbeddingService
from api.services.vector_store_service import VectorStoreService


class RagServiceError(RuntimeError):
    """Base error for retrieval orchestration failures."""


class RagService:
    """Coordinate query embedding, vector search and score filtering."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.settings = settings or load_settings()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Embed a query and return its highest-scoring knowledge chunks."""

        if not isinstance(query, str):
            raise TypeError("RAG query must be a string.")

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("RAG query cannot be empty.")

        if self.vector_store_service.index_info is None:
            self.vector_store_service.load_index()

        query_embedding = await self.embedding_service.embed_text(normalized_query)
        results = self.vector_store_service.search(
            query_embedding,
            top_k=top_k,
        )

        if self.settings.rag_min_score is not None:
            results = [
                result
                for result in results
                if result.score >= self.settings.rag_min_score
            ]

        index_info = self.vector_store_service.index_info

        return RetrievalResult(
            query=normalized_query,
            results=results,
            index_version=(index_info.format_version if index_info else None),
        )
