"""Unit and persistence tests for the local NumPy vector store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from api.config import Settings
from api.schemas.rag import KnowledgeChunk
from api.services.vector_store_service import (
    VectorStoreCorruptedError,
    VectorStoreNotFoundError,
    VectorStoreService,
    VectorStoreValidationError,
)


def make_chunks(count: int = 3) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id=f"source-section-{index:03d}",
            source="source.md",
            title="Source",
            section=f"Section {index}",
            content=f"Content {index}",
            chunk_index=index,
        )
        for index in range(1, count + 1)
    ]


def make_settings(storage_path: Path) -> Settings:
    return Settings(
        rag_storage_path=storage_path,
        rag_default_top_k=2,
        rag_max_top_k=2,
    )


def test_build_load_and_search_persisted_index(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "rag")
    store = VectorStoreService(settings=settings)
    chunks = make_chunks()

    info = store.build_index(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )

    assert info.chunk_count == 3
    assert info.document_count == 1
    assert info.embedding_dimension == 2
    assert (tmp_path / "rag" / "embeddings.npz").is_file()
    assert (tmp_path / "rag" / "metadata.json").is_file()
    assert (tmp_path / "rag" / "index_info.json").is_file()

    reloaded = VectorStoreService(settings=settings)
    loaded_info = reloaded.load_index()
    results = reloaded.search([1.0, 0.0], top_k=10)

    assert loaded_info == info.model_copy(update={"created_at": info.created_at})
    assert [result.chunk_id for result in results] == [
        "source-section-001",
        "source-section-003",
    ]
    assert all(isinstance(result.score, float) for result in results)
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)


def test_rebuilding_replaces_metadata_instead_of_appending(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "rag")
    store = VectorStoreService(settings=settings)
    chunks = make_chunks()

    store.build_index(
        chunks,
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )
    store.build_index(
        chunks[:1],
        [[0.0, 1.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )

    payload = json.loads((tmp_path / "rag" / "metadata.json").read_text())
    loaded = VectorStoreService(settings=settings)
    info = loaded.load_index()

    assert len(payload["chunks"]) == 1
    assert info.chunk_count == 1
    assert [result.chunk_id for result in loaded.search([0.0, 1.0])] == [
        "source-section-001"
    ]


def test_empty_index_is_loadable_and_returns_no_results(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "rag")
    store = VectorStoreService(settings=settings)

    info = store.build_index(
        [],
        [],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )
    reloaded = VectorStoreService(settings=settings)

    assert info.chunk_count == 0
    assert reloaded.load_index().embedding_dimension == 0
    assert reloaded.search([1.0, 0.0]) == []


def test_missing_index_is_reported(tmp_path: Path) -> None:
    store = VectorStoreService(settings=make_settings(tmp_path / "rag"))

    with pytest.raises(VectorStoreNotFoundError):
        store.load_index()


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    store = VectorStoreService(settings=make_settings(tmp_path / "rag"))
    store.build_index(
        make_chunks(1),
        [[1.0, 0.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )

    with pytest.raises(VectorStoreValidationError, match="dimension"):
        store.search([1.0, 0.0, 0.0])


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    store = VectorStoreService(settings=make_settings(tmp_path / "rag"))
    chunks = make_chunks(2)
    chunks[1] = chunks[1].model_copy(update={"chunk_id": chunks[0].chunk_id})

    with pytest.raises(VectorStoreValidationError, match="unique"):
        store.build_index(
            chunks,
            [[1.0, 0.0], [0.0, 1.0]],
            embedding_model="test-embedding",
            chunk_size=1000,
            chunk_overlap=100,
        )


def test_corrupted_metadata_is_reported(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "rag")
    store = VectorStoreService(settings=settings)
    store.build_index(
        make_chunks(1),
        [[1.0, 0.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )
    (tmp_path / "rag" / "metadata.json").write_text(
        "not json",
        encoding="utf-8",
    )

    with pytest.raises(VectorStoreCorruptedError):
        VectorStoreService(settings=settings).load_index()


def test_search_rejects_zero_query_vector(tmp_path: Path) -> None:
    store = VectorStoreService(settings=make_settings(tmp_path / "rag"))
    store.build_index(
        make_chunks(1),
        [[1.0, 0.0]],
        embedding_model="test-embedding",
        chunk_size=1000,
        chunk_overlap=100,
    )

    with pytest.raises(VectorStoreValidationError, match="non-zero"):
        store.search([0.0, 0.0])
