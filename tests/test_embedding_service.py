"""Unit tests for batched embedding validation and normalization."""

from __future__ import annotations

import asyncio
import math

import pytest

from api.services.embedding_service import (
    EmbeddingService,
    EmbeddingValidationError,
)


class FakeOllama:
    """Deterministic embedding double that records each batch."""

    def __init__(self, vectors_by_batch):
        self.vectors_by_batch = list(vectors_by_batch)
        self.calls: list[tuple[list[str], str | None]] = []

    async def embed(self, inputs, *, model=None):
        batch = list(inputs)
        self.calls.append((batch, model))
        return self.vectors_by_batch.pop(0)


def test_embed_texts_batches_inputs_and_normalizes_vectors() -> None:
    ollama = FakeOllama(
        [
            [[3.0, 4.0], [0.0, 2.0]],
            [[1.0, 1.0]]
        ]
    )
    service = EmbeddingService(
        ollama,
        embedding_model="test-embedding",
        batch_size=2,
    )

    result = asyncio.run(service.embed_texts([" first ", "second", "third"]))

    assert ollama.calls == [
        (["first", "second"], "test-embedding"),
        (["third"], "test-embedding"),
    ]
    assert result[0] == pytest.approx([0.6, 0.8])
    assert result[1] == pytest.approx([0.0, 1.0])
    assert result[2] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])


def test_embed_text_returns_one_vector() -> None:
    service = EmbeddingService(
        FakeOllama([[[0.0, 5.0]]]),
        embedding_model="test-embedding",
    )

    result = asyncio.run(service.embed_text("question"))

    assert result == [0.0, 1.0]


@pytest.mark.parametrize(
    "value, error_type",
    [
        ([], ValueError),
        ("   ", ValueError),
        (["ok", ""], ValueError),
        (["ok", 1], TypeError),
        (b"bytes", TypeError),
    ],
)
def test_embed_texts_rejects_invalid_inputs(value, error_type) -> None:
    service = EmbeddingService(FakeOllama([]), embedding_model="test-embedding")

    with pytest.raises(error_type):
        asyncio.run(service.embed_texts(value))


@pytest.mark.parametrize(
    "vectors",
    [
        [[]],
        [[0.0, 0.0]],
        [[1.0, float("nan")]],
        [[1.0, float("inf")]],
        [[1.0, "invalid"]],
        [[1.0], [1.0, 2.0]],
    ],
)
def test_embed_texts_rejects_invalid_vectors(vectors) -> None:
    service = EmbeddingService(
        FakeOllama([vectors]),
        embedding_model="test-embedding",
    )

    with pytest.raises(EmbeddingValidationError):
        asyncio.run(service.embed_texts(["question"]))


def test_embed_texts_rejects_wrong_response_count() -> None:
    service = EmbeddingService(
        FakeOllama([[[1.0, 0.0]]]),
        embedding_model="test-embedding",
    )

    with pytest.raises(EmbeddingValidationError, match="different number"):
        asyncio.run(service.embed_texts(["first", "second"]))


def test_embedding_service_propagates_transport_errors() -> None:
    class FailingOllama:
        async def embed(self, inputs, *, model=None):
            raise RuntimeError("transport failure")

    service = EmbeddingService(FailingOllama(), embedding_model="test-embedding")

    with pytest.raises(RuntimeError, match="transport failure"):
        asyncio.run(service.embed_text("question"))
