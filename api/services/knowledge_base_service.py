"""Service for loading the curated MotoStock AI knowledge base."""

from __future__ import annotations

from pathlib import Path

from api.config import Settings, load_settings
from api.rag.chunking import (
    ALLOWED_KNOWLEDGE_FILES,
    chunk_markdown_document,
)
from api.schemas.rag import KnowledgeChunk


class KnowledgeBaseServiceError(RuntimeError):
    """Base error raised by the knowledge base service."""


class KnowledgeBaseNotFoundError(KnowledgeBaseServiceError):
    """Raised when the knowledge base directory is unavailable."""


class KnowledgeDocumentMissingError(KnowledgeBaseServiceError):
    """Raised when a required knowledge document is missing."""


class KnowledgeDocumentInvalidError(KnowledgeBaseServiceError):
    """Raised when a knowledge document has invalid content."""


class KnowledgeBaseService:
    """Load and chunk the curated MotoStock AI knowledge documents."""

    def __init__(
        self,
        knowledge_base_dir: Path | None = None,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.knowledge_base_dir = (
            knowledge_base_dir or self.settings.rag_knowledge_base_path
        ).resolve()

        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else self.settings.rag_chunk_size
        )

        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else self.settings.rag_chunk_overlap
        )

        self._validate_configuration()

    def list_documents(self) -> list[Path]:
        """Return approved Markdown documents in deterministic order."""

        if not self.knowledge_base_dir.is_dir():
            raise KnowledgeBaseNotFoundError(
                "The knowledge base directory is unavailable."
            )

        document_paths = [
            self.knowledge_base_dir / filename for filename in ALLOWED_KNOWLEDGE_FILES
        ]

        missing_documents = [path.name for path in document_paths if not path.is_file()]

        if missing_documents:
            missing_names = ", ".join(sorted(missing_documents))

            raise KnowledgeDocumentMissingError(
                f"Required knowledge documents are missing: {missing_names}"
            )

        return sorted(
            document_paths,
            key=lambda path: path.name.casefold(),
        )

    def load_chunks(self) -> list[KnowledgeChunk]:
        """Load approved documents and return deterministic chunks."""

        chunks: list[KnowledgeChunk] = []
        seen_content_hashes: set[str] = set()

        for document_path in self.list_documents():
            markdown = self._read_document(document_path)

            document_chunks = chunk_markdown_document(
                source=document_path.name,
                markdown=markdown,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                seen_content_hashes=seen_content_hashes,
            )

            if not document_chunks:
                raise KnowledgeDocumentInvalidError(
                    "A knowledge document did not generate any chunks: "
                    f"{document_path.name}"
                )

            chunks.extend(document_chunks)

        return chunks

    def _read_document(self, document_path: Path) -> str:
        """Read and validate one UTF-8 Markdown document."""

        try:
            content = document_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise KnowledgeDocumentInvalidError(
                f"A knowledge document is not valid UTF-8: {document_path.name}"
            ) from exc
        except OSError as exc:
            raise KnowledgeBaseServiceError(
                f"A knowledge document could not be read: {document_path.name}"
            ) from exc

        normalized_content = content.strip()

        if not normalized_content:
            raise KnowledgeDocumentInvalidError(
                f"A knowledge document is empty: {document_path.name}"
            )

        return normalized_content

    def _validate_configuration(self) -> None:
        """Validate character-based chunk configuration."""

        if self.chunk_size <= 0:
            raise ValueError("RAG chunk size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("RAG chunk overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG chunk overlap must be smaller than chunk size.")
