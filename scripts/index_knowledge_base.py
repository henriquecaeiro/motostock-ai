"""Build the local RAG vector index from the curated knowledge base."""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from api.config import load_settings
from api.services.embedding_service import EmbeddingService
from api.services.knowledge_base_service import KnowledgeBaseService
from api.services.ollama_service import (
    OllamaConnectionError,
    OllamaModelUnavailableError,
    OllamaRequestError,
    OllamaService,
    OllamaTimeoutError,
)
from api.services.vector_store_service import VectorStoreService


async def index_knowledge_base() -> None:
    """Rebuild the local vector index once, replacing any previous version."""

    settings = load_settings()
    ollama = OllamaService(settings=settings)

    try:
        await ollama.ensure_model_available(settings.ollama_embedding_model)

        knowledge_base = KnowledgeBaseService(settings=settings)
        chunks = knowledge_base.load_chunks()

        embedding_service = EmbeddingService(
            ollama,
            settings=settings,
        )
        embeddings = await embedding_service.embed_texts(
            [chunk.content for chunk in chunks]
        )

        vector_store = VectorStoreService(settings=settings)
        index_info = vector_store.rebuild_index(
            chunks,
            embeddings,
            embedding_model=embedding_service.embedding_model,
            chunk_size=knowledge_base.chunk_size,
            chunk_overlap=knowledge_base.chunk_overlap,
        )
    finally:
        await ollama.close()

    print(f"Documents: {index_info.document_count}")
    print(f"Chunks: {index_info.chunk_count}")
    print(f"Embedding dimension: {index_info.embedding_dimension}")
    print(f"Embedding model: {index_info.embedding_model}")
    print(f"Storage path: {settings.rag_storage_path}")
    print("Index replacement: complete")


def main() -> int:
    """Run the indexing command with UTF-8-safe, concise CLI errors."""

    _configure_utf8_output()
    load_dotenv()

    try:
        asyncio.run(index_knowledge_base())
    except OllamaConnectionError:
        print("RAG indexing failed: Ollama is unavailable.", file=sys.stderr)
        return 1
    except OllamaTimeoutError:
        print("RAG indexing failed: Ollama timed out.", file=sys.stderr)
        return 1
    except OllamaModelUnavailableError as exc:
        print(
            f"RAG indexing failed: embedding model is unavailable ({exc.model}).",
            file=sys.stderr,
        )
        return 1
    except OllamaRequestError:
        print("RAG indexing failed: Ollama returned an error.", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError):
        print("RAG indexing failed: configuration or index validation error.", file=sys.stderr)
        return 1

    return 0


def _configure_utf8_output() -> None:
    """Prefer UTF-8 on Windows terminals without changing application data."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
