"""Tests for the curated knowledge base service."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.rag.chunking import ALLOWED_KNOWLEDGE_FILES
from api.schemas.rag import KnowledgeChunk
from api.services.knowledge_base_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseService,
    KnowledgeDocumentInvalidError,
    KnowledgeDocumentMissingError,
)


def create_knowledge_base(
    tmp_path: Path,
    *,
    extra_files: dict[str, str | bytes] | None = None,
    include_required: bool = True,
) -> Path:
    """Create a temporary knowledge base directory."""

    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()

    if include_required:
        for filename in ALLOWED_KNOWLEDGE_FILES:
            title = Path(filename).stem.replace("_", " ").title()
            content = (
                f"# {title}\n\n"
                f"## Overview\n\n"
                f"Minimal content for {filename}.\n"
            )
            (knowledge_base_dir / filename).write_text(
                content,
                encoding="utf-8",
            )

    if extra_files:
        for relative_path, content in extra_files.items():
            file_path = knowledge_base_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content, encoding="utf-8")

    return knowledge_base_dir


def test_list_documents_finds_all_allowed_documents(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    documents = service.list_documents()

    assert len(documents) == len(ALLOWED_KNOWLEDGE_FILES)
    assert [path.name for path in documents] == sorted(
        ALLOWED_KNOWLEDGE_FILES,
        key=str.casefold,
    )


def test_list_documents_returns_deterministic_order(tmp_path: Path) -> None:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()

    for filename in reversed(ALLOWED_KNOWLEDGE_FILES):
        (knowledge_base_dir / filename).write_text(
            f"# {filename}\n\n## Overview\n\nContent.\n",
            encoding="utf-8",
        )

    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    first_run = service.list_documents()
    second_run = service.list_documents()

    assert [path.name for path in first_run] == [path.name for path in second_run]


def test_list_documents_ignores_readme(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(
        tmp_path,
        extra_files={"README.md": "# README\n\nShould not be indexed.\n"},
    )
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    documents = service.list_documents()

    assert all(path.name != "README.md" for path in documents)


def test_load_chunks_ignores_csv_files(tmp_path: Path) -> None:
    csv_marker = "CSV_MARKER_SHOULD_NOT_BE_INDEXED"

    knowledge_base_dir = create_knowledge_base(
        tmp_path,
        extra_files={
            "predictions_comparison.csv": f"product,prediction\n{csv_marker},42\n",
        },
    )
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(not chunk.source.endswith(".csv") for chunk in chunks)
    assert all(csv_marker not in chunk.content for chunk in chunks)


def test_load_chunks_ignores_notebooks(tmp_path: Path) -> None:
    notebook_marker = "NOTEBOOK_MARKER_SHOULD_NOT_BE_INDEXED"

    knowledge_base_dir = create_knowledge_base(
        tmp_path,
        extra_files={"analysis.ipynb": f'{{"cells": ["{notebook_marker}"]}}'},
    )
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(not chunk.source.endswith(".ipynb") for chunk in chunks)
    assert all(notebook_marker not in chunk.content for chunk in chunks)


def test_load_chunks_ignores_pickle_files(tmp_path: Path) -> None:
    pickle_marker = "PKL_MARKER_SHOULD_NOT_BE_INDEXED"

    knowledge_base_dir = create_knowledge_base(
        tmp_path,
        extra_files={"model.pkl": pickle_marker},
    )
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(not chunk.source.endswith(".pkl") for chunk in chunks)
    assert all(pickle_marker not in chunk.content for chunk in chunks)


def test_load_chunks_ignores_python_source_files(tmp_path: Path) -> None:
    source_marker = "PYTHON_SOURCE_SHOULD_NOT_BE_INDEXED"

    knowledge_base_dir = create_knowledge_base(
        tmp_path,
        extra_files={"module.py": f"print('{source_marker}')"},
    )
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(not chunk.source.endswith(".py") for chunk in chunks)
    assert all(source_marker not in chunk.content for chunk in chunks)


def test_missing_knowledge_base_directory_raises_error(tmp_path: Path) -> None:
    service = KnowledgeBaseService(
        knowledge_base_dir=tmp_path / "missing-knowledge-base",
    )

    with pytest.raises(KnowledgeBaseNotFoundError):
        service.list_documents()


def test_missing_required_document_raises_error(tmp_path: Path) -> None:
    knowledge_base_dir = tmp_path / "knowledge_base"
    knowledge_base_dir.mkdir()

    (knowledge_base_dir / "business_problem.md").write_text(
        "# Business Problem\n\n## Overview\n\nPartial base.\n",
        encoding="utf-8",
    )

    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    with pytest.raises(KnowledgeDocumentMissingError):
        service.load_chunks()


def test_empty_document_raises_error(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    (knowledge_base_dir / "business_problem.md").write_text("", encoding="utf-8")

    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    with pytest.raises(KnowledgeDocumentInvalidError):
        service.load_chunks()


def test_whitespace_only_document_raises_error(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    (knowledge_base_dir / "business_problem.md").write_text(
        "   \n\n   ",
        encoding="utf-8",
    )

    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    with pytest.raises(KnowledgeDocumentInvalidError):
        service.load_chunks()


def test_invalid_utf8_document_raises_error(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    (knowledge_base_dir / "business_problem.md").write_bytes(b"\xff\xfe invalid")

    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    with pytest.raises(KnowledgeDocumentInvalidError):
        service.load_chunks()


def test_load_chunks_returns_knowledge_chunk_instances(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert chunks
    assert all(isinstance(chunk, KnowledgeChunk) for chunk in chunks)


def test_load_chunks_preserves_source(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert {chunk.source for chunk in chunks} == set(ALLOWED_KNOWLEDGE_FILES)


def test_load_chunks_preserves_title(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(chunk.title for chunk in chunks)


def test_load_chunks_preserves_section(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(chunk.section for chunk in chunks)


def test_load_chunks_do_not_return_empty_content(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert all(chunk.content.strip() for chunk in chunks)


def test_load_chunks_returns_stable_chunk_ids(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    first_run = service.load_chunks()
    second_run = service.load_chunks()

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]


def test_load_chunks_returns_stable_order(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    first_run = service.load_chunks()
    second_run = service.load_chunks()

    assert [chunk.chunk_id for chunk in first_run] == [
        chunk.chunk_id for chunk in second_run
    ]
    assert [chunk.source for chunk in first_run] == [
        chunk.source for chunk in second_run
    ]
    assert [chunk.chunk_index for chunk in first_run] == [
        chunk.chunk_index for chunk in second_run
    ]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 150),
        (-1, 150),
        (1000, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_configuration_raises_value_error(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        KnowledgeBaseService(
            knowledge_base_dir=Path("knowledge_base"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_environment_variables_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "250")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "40")

    service = KnowledgeBaseService()

    assert service.chunk_size == 250
    assert service.chunk_overlap == 40


def test_invalid_environment_variable_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "not-a-number")

    with pytest.raises(ValueError, match="RAG_CHUNK_SIZE must be a valid integer"):
        KnowledgeBaseService()


def test_load_chunks_do_not_include_embeddings(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    for chunk in chunks:
        dumped = chunk.model_dump()
        assert "embedding" not in dumped
        assert "vector" not in dumped
        assert "similarity" not in dumped
        assert "score" not in dumped


def test_load_chunks_does_not_depend_on_ollama_service(tmp_path: Path) -> None:
    knowledge_base_dir = create_knowledge_base(tmp_path)
    service = KnowledgeBaseService(knowledge_base_dir=knowledge_base_dir)

    chunks = service.load_chunks()

    assert chunks
    assert all(isinstance(chunk, KnowledgeChunk) for chunk in chunks)
