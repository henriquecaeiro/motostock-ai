"""Run representative manual retrieval checks against the local RAG index."""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from api.config import load_settings
from api.services.embedding_service import EmbeddingService
from api.services.ollama_service import OllamaService
from api.services.rag_service import RagService
from api.services.vector_store_service import VectorStoreService


CHECK_QUERIES = (
    "Why was XGBoost selected?",
    "Como o estoque de segurança é calculado?",
    "What are the limitations of the demand forecast?",
    "Qual é a responsabilidade da linguagem natural no MotoStock?",
    "How is a critical stock status determined?",
    "Who won the football championship?",
)


async def check_retrieval() -> None:
    """Print ranked sources and previews for representative questions."""

    settings = load_settings()
    ollama = OllamaService(settings=settings)

    try:
        await ollama.ensure_model_available(settings.ollama_embedding_model)
        vector_store = VectorStoreService(settings=settings)
        vector_store.load_index()
        embedding_service = EmbeddingService(ollama, settings=settings)
        rag_service = RagService(
            embedding_service,
            vector_store,
            settings=settings,
        )

        for query in CHECK_QUERIES:
            result = await rag_service.retrieve(query)
            print(f"\nQuery: {query}")

            if not result.results:
                print("  No results above the configured threshold.")
                continue

            for rank, chunk in enumerate(result.results, start=1):
                preview = " ".join(chunk.content.split())[:180]
                print(
                    f"  {rank}. score={chunk.score:.4f} "
                    f"source={chunk.source} section={chunk.section}\n"
                    f"     {preview}"
                )
    finally:
        await ollama.close()


def main() -> int:
    """Run the manual retrieval check with concise errors."""

    _configure_utf8_output()
    load_dotenv()

    try:
        asyncio.run(check_retrieval())
    except Exception:
        print(
            "RAG retrieval check failed: verify Ollama, the embedding model and the local index.",
            file=sys.stderr,
        )
        return 1

    return 0


def _configure_utf8_output() -> None:
    """Prefer UTF-8 on Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
