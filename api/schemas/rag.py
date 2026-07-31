"""Schemas used by the Retrieval-Augmented Generation pipeline"""

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
