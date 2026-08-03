"""Schemas used by the Retrieval-Augmented Generation pipeline."""

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    """A deterministic chunk extracted from a knowledge document."""

    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier generated from the source and section",
    )

    source: str = Field(
        ..., min_length=1, description="Name of the Markdown source file"
    )

    title: str = Field(
        ...,
        min_length=1,
        description="Main title of the source document",
    )

    section: str = Field(
        ...,
        min_length=1,
        description="Markdown section represented by the chunk",
    )

    content: str = Field(
        ...,
        min_length=1,
        description="Text that will be transformed into an embedding",
    )

    chunk_index: int = Field(
        ...,
        ge=1,
        description="One-based position of the chunk inside the source file",
    )


class RetrievedChunk(BaseModel):
    """A knowledge chunk together with its cosine similarity score."""

    chunk_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=1)
    score: float = Field(..., ge=-1.0, le=1.0)


class RetrievalResult(BaseModel):
    """Structured output from a semantic knowledge-base search."""

    query: str = Field(..., min_length=1)
    results: list[RetrievedChunk] = Field(default_factory=list)
    index_version: int | None = Field(default=None, ge=1)


class RagIndexInfo(BaseModel):
    """Metadata needed to validate and describe a persisted RAG index."""

    format_version: int = Field(default=1, ge=1)
    created_at: str = Field(..., min_length=1)
    embedding_model: str = Field(..., min_length=1)
    embedding_dimension: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    document_count: int = Field(..., ge=0)
    chunk_size: int = Field(..., ge=1)
    chunk_overlap: int = Field(..., ge=0)
    sources: list[str] = Field(default_factory=list)
