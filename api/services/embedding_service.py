"""Validated, batched embedding generation for the RAG pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence

from api.config import Settings, load_settings
from api.services.ollama_service import OllamaService


class EmbeddingServiceError(RuntimeError):
    """Base error raised by the embedding service."""


class EmbeddingValidationError(EmbeddingServiceError):
    """Raised when an embedding response is malformed or inconsistent."""


class EmbeddingService:
    """Generate normalized embeddings through one shared Ollama client.

    Ollama failures are intentionally allowed to propagate unchanged so API
    callers can map connection, timeout and model errors to safe HTTP responses.
    Response-shape and vector validation failures use this service's error type.
    """

    def __init__(
        self,
        ollama_service: OllamaService,
        *,
        embedding_model: str | None = None,
        batch_size: int = 32,
        settings: Settings | None = None,
    ) -> None:
        self.ollama_service = ollama_service
        self.settings = settings or load_settings()
        self.embedding_model = (
            embedding_model or self.settings.ollama_embedding_model
        ).strip()
        self.batch_size = batch_size

        if not self.embedding_model:
            raise ValueError("The embedding model cannot be empty.")

        if self.batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero.")

    async def embed_text(self, text: str) -> list[float]:
        """Generate one normalized embedding."""

        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: str | Sequence[str]) -> list[list[float]]:
        """Generate normalized embeddings in bounded batches."""

        normalized_texts = self._normalize_texts(texts)
        all_vectors: list[list[float]] = []
        expected_dimension: int | None = None

        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            vectors = await self.ollama_service.embed(
                batch,
                model=self.embedding_model,
            )

            if len(vectors) != len(batch):
                raise EmbeddingValidationError(
                    "Ollama returned a different number of embeddings than inputs."
                )

            normalized_vectors = self._normalize_vectors(vectors)

            if expected_dimension is None:
                expected_dimension = len(normalized_vectors[0])
            elif any(
                len(vector) != expected_dimension
                for vector in normalized_vectors
            ):
                raise EmbeddingValidationError(
                    "Embedding batches returned inconsistent dimensions."
                )

            all_vectors.extend(normalized_vectors)

        return all_vectors

    @staticmethod
    def _normalize_texts(texts: str | Sequence[str]) -> list[str]:
        """Validate and trim input strings before sending them to Ollama."""

        if isinstance(texts, str):
            values = [texts]
        elif isinstance(texts, (bytes, bytearray)):
            raise TypeError("Embedding inputs must be strings.")
        elif isinstance(texts, Sequence):
            values = list(texts)
        else:
            raise TypeError("Embedding inputs must be a string or a sequence of strings.")

        if not values:
            raise ValueError("At least one embedding input is required.")

        normalized: list[str] = []

        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise TypeError(
                    f"Embedding input at position {index} must be a string."
                )

            stripped = value.strip()

            if not stripped:
                raise ValueError(
                    f"Embedding input at position {index} cannot be empty."
                )

            normalized.append(stripped)

        return normalized

    @staticmethod
    def _normalize_vectors(
        vectors: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Validate finite vectors and convert each one to unit length."""

        if not isinstance(vectors, Sequence) or not vectors:
            raise EmbeddingValidationError("Embedding response cannot be empty.")

        normalized_vectors: list[list[float]] = []
        expected_dimension: int | None = None

        for index, vector in enumerate(vectors):
            if isinstance(vector, (str, bytes, bytearray)) or not isinstance(
                vector,
                Sequence,
            ):
                raise EmbeddingValidationError(
                    f"Embedding at position {index} is not a sequence."
                )

            if not vector:
                raise EmbeddingValidationError(
                    f"Embedding at position {index} is empty."
                )

            converted: list[float] = []

            for value in vector:
                if isinstance(value, bool):
                    raise EmbeddingValidationError(
                        f"Embedding at position {index} contains a boolean value."
                    )

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise EmbeddingValidationError(
                        f"Embedding at position {index} contains a non-numeric value."
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise EmbeddingValidationError(
                        f"Embedding at position {index} contains a non-finite value."
                    )

                converted.append(numeric_value)

            dimension = len(converted)

            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise EmbeddingValidationError(
                    "Embedding vectors have inconsistent dimensions."
                )

            norm = math.sqrt(sum(value * value for value in converted))

            if not math.isfinite(norm) or norm == 0:
                raise EmbeddingValidationError(
                    f"Embedding at position {index} has zero magnitude."
                )

            normalized_vectors.append([value / norm for value in converted])

        return normalized_vectors
