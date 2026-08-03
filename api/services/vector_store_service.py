"""Small, local NumPy vector store used by MotoStock AI RAG."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from api.config import Settings, load_settings
from api.schemas.rag import KnowledgeChunk, RagIndexInfo, RetrievedChunk


class VectorStoreError(RuntimeError):
    """Base error raised by the local vector store."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when a complete persisted index is not available."""


class VectorStoreCorruptedError(VectorStoreError):
    """Raised when persisted vector data or metadata is inconsistent."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when an index build or query has invalid values."""


class VectorStoreService:
    """Persist and search normalized embeddings with cosine similarity."""

    FORMAT_VERSION = 1
    EMBEDDINGS_FILENAME = "embeddings.npz"
    METADATA_FILENAME = "metadata.json"
    INDEX_INFO_FILENAME = "index_info.json"

    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.storage_path = (
            storage_path or self.settings.rag_storage_path
        ).resolve()
        self.embeddings_path = self.storage_path / self.EMBEDDINGS_FILENAME
        self.metadata_path = self.storage_path / self.METADATA_FILENAME
        self.index_info_path = self.storage_path / self.INDEX_INFO_FILENAME

        self._embeddings: np.ndarray | None = None
        self._chunks: list[KnowledgeChunk] | None = None
        self._index_info: RagIndexInfo | None = None

    @property
    def has_persisted_index(self) -> bool:
        """Whether all files required for a complete index are present."""

        return all(
            path.is_file()
            for path in (
                self.embeddings_path,
                self.metadata_path,
                self.index_info_path,
            )
        )

    @property
    def index_info(self) -> RagIndexInfo | None:
        """Return loaded metadata, if the index has already been loaded."""

        return self._index_info

    def rebuild_index(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> RagIndexInfo:
        """Replace the complete index with the supplied chunks and vectors."""

        return self.build_index(
            chunks,
            embeddings,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build_index(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> RagIndexInfo:
        """Validate and atomically persist a complete replacement index."""

        chunk_list = list(chunks)
        embedding_matrix = self._prepare_embeddings(embeddings)
        self._validate_build_inputs(
            chunks=chunk_list,
            embeddings=embedding_matrix,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        info = RagIndexInfo(
            format_version=self.FORMAT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=embedding_model.strip(),
            embedding_dimension=(
                int(embedding_matrix.shape[1])
                if embedding_matrix.ndim == 2
                else 0
            ),
            chunk_count=len(chunk_list),
            document_count=len({chunk.source for chunk in chunk_list}),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sources=sorted({chunk.source for chunk in chunk_list}),
        )

        metadata = {
            "chunks": [self._model_dump(chunk) for chunk in chunk_list],
        }

        self.storage_path.mkdir(parents=True, exist_ok=True)

        try:
            self._atomic_write_npz(embedding_matrix)
            self._atomic_write_json(self.metadata_path, metadata)
            self._atomic_write_json(self.index_info_path, self._model_dump(info))
        except Exception as exc:
            raise VectorStoreError("The RAG index could not be persisted.") from exc

        self._embeddings = embedding_matrix
        self._chunks = chunk_list
        self._index_info = info

        return info

    def load_index(self) -> RagIndexInfo:
        """Load and validate all persisted index files."""

        if not self.has_persisted_index:
            raise VectorStoreNotFoundError(
                "The RAG vector index is not available."
            )

        try:
            with np.load(self.embeddings_path, allow_pickle=False) as payload:
                if "embeddings" not in payload:
                    raise VectorStoreCorruptedError(
                        "The RAG embeddings file has no embeddings array."
                    )

                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)

            metadata_payload = self._read_json(self.metadata_path)
            index_info_payload = self._read_json(self.index_info_path)
            info = RagIndexInfo(**index_info_payload)
            chunks = self._parse_chunks(metadata_payload)
            self._validate_loaded_index(
                embeddings=embeddings,
                chunks=chunks,
                info=info,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreCorruptedError(
                "The RAG vector index is corrupted or invalid."
            ) from exc

        self._embeddings = embeddings
        self._chunks = chunks
        self._index_info = info

        return info

    def clear_loaded_index(self) -> None:
        """Release in-memory vectors without deleting persisted files."""

        self._embeddings = None
        self._chunks = None
        self._index_info = None

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the highest-scoring chunks for one normalized query vector."""

        if self._embeddings is None or self._chunks is None or self._index_info is None:
            self.load_index()

        assert self._embeddings is not None
        assert self._chunks is not None
        assert self._index_info is not None

        if not self._chunks:
            return []

        query = self._normalize_query_embedding(query_embedding)

        if query.shape[0] != self._index_info.embedding_dimension:
            raise VectorStoreValidationError(
                "Query embedding dimension does not match the RAG index."
            )

        requested_top_k = self._normalize_top_k(top_k)
        scores = self._embeddings @ query
        ordered_indexes = np.argsort(-scores, kind="stable")[:requested_top_k]

        return [
            RetrievedChunk(
                **self._model_dump(self._chunks[int(index)]),
                score=float(np.clip(scores[int(index)], -1.0, 1.0)),
            )
            for index in ordered_indexes
        ]

    def _normalize_top_k(self, top_k: int | None) -> int:
        """Validate a requested top-k and enforce the configured maximum."""

        selected_top_k = (
            self.settings.rag_default_top_k if top_k is None else top_k
        )

        if not isinstance(selected_top_k, int) or isinstance(selected_top_k, bool):
            raise ValueError("top_k must be an integer.")

        if selected_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return min(selected_top_k, self.settings.rag_max_top_k)

    @staticmethod
    def _prepare_embeddings(
        embeddings: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Convert vectors to a finite, row-normalized float32 matrix."""

        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VectorStoreValidationError(
                "Embeddings must be a rectangular numeric matrix."
            ) from exc

        if matrix.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise VectorStoreValidationError(
                "Embeddings must be a non-empty two-dimensional matrix."
            )

        if not np.isfinite(matrix).all():
            raise VectorStoreValidationError("Embeddings must contain finite values.")

        norms = np.linalg.norm(matrix, axis=1)

        if np.any(~np.isfinite(norms)) or np.any(norms == 0):
            raise VectorStoreValidationError(
                "Embeddings must have non-zero finite magnitude."
            )

        return (matrix / norms[:, np.newaxis]).astype(np.float32)

    @staticmethod
    def _normalize_query_embedding(query_embedding: Sequence[float]) -> np.ndarray:
        """Validate and normalize one query vector."""

        try:
            query = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VectorStoreValidationError(
                "Query embedding must be a numeric vector."
            ) from exc

        if query.ndim != 1 or query.size == 0:
            raise VectorStoreValidationError(
                "Query embedding must be a non-empty one-dimensional vector."
            )

        if not np.isfinite(query).all():
            raise VectorStoreValidationError(
                "Query embedding must contain finite values."
            )

        norm = float(np.linalg.norm(query))

        if not math.isfinite(norm) or norm == 0:
            raise VectorStoreValidationError(
                "Query embedding must have non-zero magnitude."
            )

        return query / norm

    def _validate_build_inputs(
        self,
        *,
        chunks: list[KnowledgeChunk],
        embeddings: np.ndarray,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Validate the relationship between chunks, vectors and settings."""

        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise VectorStoreValidationError("Embedding model cannot be empty.")

        if chunk_size <= 0:
            raise VectorStoreValidationError("Chunk size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise VectorStoreValidationError(
                "Chunk overlap must be non-negative and smaller than chunk size."
            )

        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreValidationError(
                "The number of chunks must match the number of embeddings."
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorStoreValidationError("Chunk IDs must be unique.")

        if any(not chunk.content.strip() for chunk in chunks):
            raise VectorStoreValidationError("Chunk content cannot be empty.")

    def _validate_loaded_index(
        self,
        *,
        embeddings: np.ndarray,
        chunks: list[KnowledgeChunk],
        info: RagIndexInfo,
    ) -> None:
        """Validate persisted counts, dimensions and metadata relationships."""

        if info.format_version != self.FORMAT_VERSION:
            raise VectorStoreCorruptedError(
                "The RAG index format version is not supported."
            )

        if embeddings.size == 0:
            embeddings = np.empty((0, 0), dtype=np.float32)

        if embeddings.ndim != 2:
            raise VectorStoreCorruptedError(
                "The RAG embeddings array must be two-dimensional."
            )

        if not np.isfinite(embeddings).all():
            raise VectorStoreCorruptedError(
                "The RAG embeddings array contains non-finite values."
            )

        if len(chunks) != info.chunk_count or embeddings.shape[0] != info.chunk_count:
            raise VectorStoreCorruptedError(
                "RAG index chunk counts are inconsistent."
            )

        if embeddings.shape[1] != info.embedding_dimension:
            raise VectorStoreCorruptedError(
                "RAG index embedding dimensions are inconsistent."
            )

        if info.chunk_count > 0 and info.embedding_dimension == 0:
            raise VectorStoreCorruptedError(
                "A non-empty RAG index must have a positive embedding dimension."
            )

        if info.chunk_count > 0:
            norms = np.linalg.norm(embeddings, axis=1)

            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
                raise VectorStoreCorruptedError(
                    "Persisted RAG embeddings are not normalized."
                )

        if len({chunk.source for chunk in chunks}) != info.document_count:
            raise VectorStoreCorruptedError(
                "RAG index document counts are inconsistent."
            )

        if sorted({chunk.source for chunk in chunks}) != sorted(info.sources):
            raise VectorStoreCorruptedError(
                "RAG index source metadata is inconsistent."
            )

        self._validate_build_inputs(
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=info.embedding_model,
            chunk_size=info.chunk_size,
            chunk_overlap=info.chunk_overlap,
        )

    @staticmethod
    def _parse_chunks(payload: dict[str, Any]) -> list[KnowledgeChunk]:
        """Parse and validate serialized chunk metadata."""

        raw_chunks = payload.get("chunks")

        if not isinstance(raw_chunks, list):
            raise VectorStoreCorruptedError(
                "The RAG metadata file has no valid chunks list."
            )

        try:
            return [KnowledgeChunk(**item) for item in raw_chunks]
        except Exception as exc:
            raise VectorStoreCorruptedError(
                "The RAG metadata file contains invalid chunks."
            ) from exc

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read one JSON object from disk."""

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise VectorStoreCorruptedError(
                f"The RAG metadata file is not a JSON object: {path.name}."
            )

        return payload

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write JSON to a sibling temporary file and replace the target."""

        temporary_path = self._temporary_path(path.suffix)

        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")

            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _atomic_write_npz(self, embeddings: np.ndarray) -> None:
        """Write the NumPy array to a sibling temporary file and replace it."""

        temporary_path = self._temporary_path(".npz")

        try:
            with temporary_path.open("wb") as file:
                np.savez_compressed(file, embeddings=embeddings)

            os.replace(temporary_path, self.embeddings_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _temporary_path(self, suffix: str) -> Path:
        """Create a unique temporary path inside the configured storage folder."""

        descriptor, path_string = tempfile.mkstemp(
            prefix=".rag-index-",
            suffix=suffix,
            dir=self.storage_path,
        )
        os.close(descriptor)
        return Path(path_string)

    @staticmethod
    def _model_dump(model: Any) -> dict[str, Any]:
        """Serialize a Pydantic model across supported Pydantic versions."""

        if hasattr(model, "model_dump"):
            return model.model_dump()

        return model.dict()
