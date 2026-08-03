"""Tests for deterministic knowledge-base chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.rag.chunking import (
    ALLOWED_KNOWLEDGE_FILES,
    build_knowledge_chunks,
    chunk_markdown_document,
)

FORBIDDEN_SOURCE_MARKERS = (
    "README.md",
    ".csv",
    ".ipynb",
    ".pkl",
    ".py",
    "__pycache__",
)


def create_minimal_knowledge_base(
    tmp_path: Path,
    *,
    extra_files: dict[str, str] | None = None,
    include_required: bool = True,
) -> Path:
    """Create a temporary knowledge base directory for loader tests."""

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
            file_path.write_text(content, encoding="utf-8")

    return knowledge_base_dir


def test_build_knowledge_chunks_only_loads_allowed_documents(tmp_path: Path) -> None:
  knowledge_base_dir = create_minimal_knowledge_base(
      tmp_path,
      extra_files={
          "README.md": "# README\n\nShould not be indexed.\n",
          "predictions_comparison.csv": "product,prediction\nHelmet,10\n",
          "sales.csv": "date,product,units\n2024-01-01,Helmet,5\n",
          "notebook.ipynb": '{"cells": []}',
          "model.pkl": "binary-model-content",
          "source.py": "print('not for rag')",
          "__pycache__/compiled.cpython-312.pyc": "bytecode",
      },
  )

  chunks = build_knowledge_chunks(knowledge_base_dir)

  sources = {chunk.source for chunk in chunks}

  assert sources
  assert sources.issubset(set(ALLOWED_KNOWLEDGE_FILES))

  for marker in FORBIDDEN_SOURCE_MARKERS:
      assert all(marker not in chunk.source for chunk in chunks)


def test_build_knowledge_chunks_does_not_index_readme(tmp_path: Path) -> None:
  readme_marker = "README_SHOULD_NEVER_APPEAR_IN_CHUNKS"

  knowledge_base_dir = create_minimal_knowledge_base(
      tmp_path,
      extra_files={
          "README.md": f"# README\n\n{readme_marker}\n",
      },
  )

  chunks = build_knowledge_chunks(knowledge_base_dir)

  assert all(chunk.source != "README.md" for chunk in chunks)
  assert all(readme_marker not in chunk.content for chunk in chunks)


def test_build_knowledge_chunks_does_not_index_csv_files(tmp_path: Path) -> None:
  csv_marker = "CSV_MARKER_SHOULD_NEVER_BE_INDEXED"

  knowledge_base_dir = create_minimal_knowledge_base(
      tmp_path,
      extra_files={
          "predictions_comparison.csv": f"product,prediction\n{csv_marker},42\n",
      },
  )

  chunks = build_knowledge_chunks(knowledge_base_dir)

  assert all(not chunk.source.endswith(".csv") for chunk in chunks)
  assert all(csv_marker not in chunk.content for chunk in chunks)


def test_chunk_markdown_document_does_not_create_empty_chunks() -> None:
  markdown = """
# Document Title

## Empty Section



## Whitespace Section

   

## Valid Section

Actual content here.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert chunks
  assert all(chunk.content.strip() for chunk in chunks)


def test_chunk_preserves_document_title() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert len(chunks) == 1
  assert chunks[0].title == "Stock Recommendation Rules"
  assert "# Stock Recommendation Rules" in chunks[0].content


def test_chunk_preserves_markdown_section() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert chunks[0].section == "Safety Stock"
  assert "## Safety Stock" in chunks[0].content


def test_chunk_preserves_source_filename() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert chunks[0].source == "stock_recommendation_rules.md"


def test_chunk_index_starts_at_one_and_is_sequential() -> None:
  long_body = " ".join(f"sentence{index:03d}." for index in range(1, 80))

  markdown = f"""
# Stock Recommendation Rules

## Safety Stock

{long_body}
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=180,
      chunk_overlap=30,
  )

  assert len(chunks) > 1
  assert [chunk.chunk_index for chunk in chunks] == list(
      range(1, len(chunks) + 1)
  )


def test_chunk_ids_are_stable_between_equal_indexing_runs() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  first_result = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=1000,
      chunk_overlap=150,
  )

  second_result = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=1000,
      chunk_overlap=150,
  )

  assert [chunk.chunk_id for chunk in first_result] == [
      chunk.chunk_id for chunk in second_result
  ]


def test_chunk_id_contains_source_section_and_stable_index() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  chunk_id = chunks[0].chunk_id

  assert "stock-recommendation-rules" in chunk_id
  assert "safety-stock" in chunk_id
  assert chunk_id.endswith("-001")


def test_build_knowledge_chunks_has_deterministic_order(tmp_path: Path) -> None:
  knowledge_base_dir = tmp_path / "knowledge_base"
  knowledge_base_dir.mkdir()

  creation_order = list(reversed(ALLOWED_KNOWLEDGE_FILES))

  for filename in creation_order:
      title = Path(filename).stem.replace("_", " ").title()
      content = (
          f"# {title}\n\n"
          f"## Overview\n\n"
          f"Deterministic content for {filename}.\n"
      )
      (knowledge_base_dir / filename).write_text(
          content,
          encoding="utf-8",
      )

  first_run = build_knowledge_chunks(knowledge_base_dir)
  second_run = build_knowledge_chunks(knowledge_base_dir)

  assert [chunk.chunk_id for chunk in first_run] == [
      chunk.chunk_id for chunk in second_run
  ]
  assert [chunk.source for chunk in first_run] == [
      chunk.source for chunk in second_run
  ]
  assert [chunk.chunk_index for chunk in first_run] == [
      chunk.chunk_index for chunk in second_run
  ]


def test_build_knowledge_chunks_removes_duplicate_chunks(tmp_path: Path) -> None:
  knowledge_base_dir = tmp_path / "knowledge_base"
  knowledge_base_dir.mkdir()

  duplicate_markdown = (
      "# Shared Title\n\n"
      "## Shared Section\n\n"
      "Exact duplicate paragraph for deduplication testing."
  )

  for filename in ALLOWED_KNOWLEDGE_FILES:
      (knowledge_base_dir / filename).write_text(
          duplicate_markdown,
          encoding="utf-8",
      )

  chunks = build_knowledge_chunks(knowledge_base_dir)

  normalized_contents = [
      " ".join(chunk.content.split())
      for chunk in chunks
  ]

  assert len(normalized_contents) == len(set(normalized_contents))


def test_long_markdown_section_is_split_into_multiple_chunks() -> None:
  long_body = " ".join(f"word{index:03d}" for index in range(1, 120))

  markdown = f"""
# Stock Recommendation Rules

## Safety Stock

{long_body}
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=180,
      chunk_overlap=30,
  )

  assert len(chunks) > 1

  for chunk in chunks:
      assert chunk.title == "Stock Recommendation Rules"
      assert chunk.section == "Safety Stock"
      assert chunk.content.strip()


def test_consecutive_chunks_preserve_configured_overlap() -> None:
  numbered_words = " ".join(f"token{index:03d}" for index in range(1, 80))

  markdown = f"""
# Stock Recommendation Rules

## Safety Stock

{numbered_words}
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=180,
      chunk_overlap=40,
  )

  assert len(chunks) > 1

  for previous_chunk, next_chunk in zip(chunks, chunks[1:]):
      previous_tokens = set(previous_chunk.content.split())
      next_tokens = set(next_chunk.content.split())
      assert previous_tokens & next_tokens


def test_chunking_avoids_cutting_words_when_boundary_is_available() -> None:
    words = [
        "inventory",
        "forecasting",
        "replenishment",
        "safety",
        "stock",
        "planning",
        "warehouse",
        "operations",
        "supplier",
        "leadtime",
    ]

    markdown = f"""
# Stock Recommendation Rules

## Safety Stock

{" ".join(words)}
""".strip()

    chunks = chunk_markdown_document(
        source="stock_recommendation_rules.md",
        markdown=markdown,
        chunk_size=60,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for word in words:
        for chunk in chunks:
            for token in chunk.content.split():
                if token == word:
                    continue

                if len(token) < len(word) and word.startswith(token):
                    pytest.fail(
                        f"Found partial word fragment '{token}' from '{word}'."
                    )

                if len(token) > len(word) and token.startswith(word):
                    pytest.fail(
                        f"Found partial word fragment '{token}' from '{word}'."
                    )


def test_chunk_content_respects_configured_size() -> None:
  long_body = " ".join(f"word{index:03d}" for index in range(1, 100))

  markdown = f"""
# Stock Recommendation Rules

## Safety Stock

{long_body}
""".strip()

  chunk_size = 250

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
      chunk_size=chunk_size,
      chunk_overlap=40,
  )

  assert chunks
  assert all(len(chunk.content) <= chunk_size for chunk in chunks)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_raises_value_error(chunk_size: int) -> None:
  with pytest.raises(ValueError, match="chunk size"):
      chunk_markdown_document(
          source="stock_recommendation_rules.md",
          markdown="# Title\n\n## Section\n\nBody.",
          chunk_size=chunk_size,
          chunk_overlap=0,
      )


def test_negative_chunk_overlap_raises_value_error() -> None:
  with pytest.raises(ValueError, match="overlap cannot be negative"):
      chunk_markdown_document(
          source="stock_recommendation_rules.md",
          markdown="# Title\n\n## Section\n\nBody.",
          chunk_size=100,
          chunk_overlap=-1,
      )


@pytest.mark.parametrize("chunk_overlap", [100, 101])
def test_overlap_must_be_smaller_than_chunk_size(chunk_overlap: int) -> None:
  with pytest.raises(ValueError, match="overlap must be smaller"):
      chunk_markdown_document(
          source="stock_recommendation_rules.md",
          markdown="# Title\n\n## Section\n\nBody.",
          chunk_size=100,
          chunk_overlap=chunk_overlap,
      )


def test_missing_knowledge_base_directory_raises_error(tmp_path: Path) -> None:
  missing_dir = tmp_path / "missing-knowledge-base"

  with pytest.raises(FileNotFoundError, match="does not exist"):
      build_knowledge_chunks(missing_dir)


def test_missing_required_knowledge_document_raises_error(tmp_path: Path) -> None:
  knowledge_base_dir = tmp_path / "knowledge_base"
  knowledge_base_dir.mkdir()

  (knowledge_base_dir / "business_problem.md").write_text(
      "# Business Problem\n\n## Overview\n\nPartial base.\n",
      encoding="utf-8",
  )

  with pytest.raises(FileNotFoundError, match="Required knowledge documents are missing"):
      build_knowledge_chunks(knowledge_base_dir)


def test_missing_h1_uses_filename_as_fallback_title() -> None:
  markdown = """
## Safety Stock

Some content.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert chunks[0].title == "Stock Recommendation Rules"


def test_h3_chunk_preserves_parent_section_context() -> None:
  markdown = """
# Stock Recommendation Rules

## Stock Status

### Critical

Critical means the product needs urgent attention.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  chunk = chunks[0]

  assert chunk.title == "Stock Recommendation Rules"
  assert chunk.section == "Critical"
  assert "## Stock Status" in chunk.content
  assert "### Critical" in chunk.content
  assert "Critical means the product needs urgent attention." in chunk.content


def test_line_ending_style_does_not_change_chunk_ids() -> None:
  markdown_lf = (
      "# Stock Recommendation Rules\n\n"
      "## Safety Stock\n\n"
      "Safety stock protects inventory against uncertainty."
  )

  markdown_crlf = markdown_lf.replace("\n", "\r\n")

  lf_chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown_lf,
  )

  crlf_chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown_crlf,
  )

  assert [chunk.chunk_id for chunk in lf_chunks] == [
      chunk.chunk_id for chunk in crlf_chunks
  ]

  assert [
      " ".join(chunk.content.split())
      for chunk in lf_chunks
  ] == [
      " ".join(chunk.content.split())
      for chunk in crlf_chunks
  ]


def test_chunking_preserves_utf8_content() -> None:
  markdown = """
# Regras de Recomendação

## Estoque de Segurança

O estoque de segurança protege contra incerteza: café, previsão, segurança, ×, ≤.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  content = chunks[0].content

  assert "café" in content
  assert "previsão" in content
  assert "segurança" in content
  assert "×" in content
  assert "≤" in content


def test_chunk_preserves_document_metadata() -> None:
  markdown = """
# Stock Recommendation Rules

## Safety Stock

Safety stock protects inventory against uncertainty.
""".strip()

  chunks = chunk_markdown_document(
      source="stock_recommendation_rules.md",
      markdown=markdown,
  )

  assert len(chunks) == 1

  chunk = chunks[0]

  assert chunk.source == "stock_recommendation_rules.md"
  assert chunk.title == "Stock Recommendation Rules"
  assert chunk.section == "Safety Stock"
  assert chunk.chunk_index == 1
  assert chunk.chunk_id
  assert "# Stock Recommendation Rules" in chunk.content
  assert "## Safety Stock" in chunk.content
  assert "Safety stock protects inventory" in chunk.content
