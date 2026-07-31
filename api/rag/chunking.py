"""Load and split curated MotoStock AI knowledge documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from api.schemas.rag import KnowledgeChunk

ALLOWED_KNOWLEDGE_FILES = (
    "business_problem.md",
    "forecasting_model.md",
    "glossary.md",
    "limitations_and_data_access.md",
    "model_evaluation.md",
    "stock_recommendation_rules.md",
    "system_architecture.md",
)

MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownSection:
    """A section extracted from a Markdown document."""

    name: str
    headings: tuple[str, ...]
    body: str


def build_knowledge_chunks(
    knowledge_base_dir: Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[KnowledgeChunk]:
    """Load approved documents and return deterministic chunks."""

    _validate_chunk_configuration(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    knowledge_base_dir = knowledge_base_dir.resolve()

    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory does not exist: {knowledge_base_dir}"
        )

    missing_files = [
        filename
        for filename in ALLOWED_KNOWLEDGE_FILES
        if not (knowledge_base_dir / filename).is_file()
    ]

    if missing_files:
        missing_names = ", ".join(sorted(missing_files))

        raise FileNotFoundError(
            f"Required knowledge documents are missing: {missing_names}"
        )

    chunks: list[KnowledgeChunk] = []
    seen_content_hashes: set[str] = set()

    document_paths = sorted(
        (knowledge_base_dir / filename for filename in ALLOWED_KNOWLEDGE_FILES),
        key=lambda path: path.name.casefold(),
    )

    for document_path in document_paths:
        document_chunks = chunk_markdown_document(
            source=document_path.name,
            markdown=document_path.read_text(encoding="utf-8"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            seen_content_hashes=seen_content_hashes,
        )

        chunks.extend(document_chunks)

    return chunks


def chunk_markdown_document(
    *,
    source: str,
    markdown: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    seen_content_hashes: set[str] | None = None,
) -> list[KnowledgeChunk]:
    """Split one Markdown document while preserving heading context."""

    _validate_chunk_configuration(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    normalized_markdown = _normalize_markdown(markdown)

    if not normalized_markdown:
        return []

    fallback_title = Path(source).stem.replace("_", " ").title()

    title, sections = _parse_markdown_sections(
        normalized_markdown,
        fallback_title=fallback_title,
    )

    chunks: list[KnowledgeChunk] = []
    content_hashes = seen_content_hashes if seen_content_hashes is not None else set()

    source_slug = _slugify(Path(source).stem)
    chunk_index = 0
    section_occurrences: dict[str, int] = {}

    for section in sections:
        section_slug = _slugify(section.name) or "overview"

        occurrence = section_occurrences.get(section_slug, 0) + 1
        section_occurrences[section_slug] = occurrence

        id_section_slug = section_slug

        if occurrence > 1:
            id_section_slug = f"{section_slug}-{occurrence}"

        prefix_lines = [
            f"# {title}",
            *section.headings,
        ]

        prefix = "\n".join(prefix_lines).strip()

        body_chunk_size = chunk_size - len(prefix) - 2

        if body_chunk_size <= 0:
            raise ValueError(
                "The configured chunk size is too small for the "
                f"document headings in {source}."
            )

        effective_overlap = min(
            chunk_overlap,
            max(0, body_chunk_size - 1),
        )

        section_part_index = 0

        for body_part in _split_text(
            section.body,
            chunk_size=body_chunk_size,
            chunk_overlap=effective_overlap,
        ):
            content = f"{prefix}\n\n{body_part}".strip()

            if not content:
                continue

            content_hash = _create_content_hash(content)

            if content_hash in content_hashes:
                continue

            content_hashes.add(content_hash)

            chunk_index += 1
            section_part_index += 1

            chunk_id = f"{source_slug}-{id_section_slug}-{section_part_index:03d}"

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    source=source,
                    title=title,
                    section=section.name,
                    content=content,
                    chunk_index=chunk_index,
                )
            )

    return chunks


def _parse_markdown_sections(
    markdown: str,
    *,
    fallback_title: str,
) -> tuple[str, list[MarkdownSection]]:
    """Extract the document title and sections from Markdown."""

    document_title = fallback_title
    sections: list[MarkdownSection] = []

    current_section_name = "Overview"
    current_headings: list[str] = []
    current_body: list[str] = []
    current_level_two_heading: str | None = None

    def flush_section() -> None:
        body = "\n".join(current_body).strip()

        if body:
            sections.append(
                MarkdownSection(
                    name=current_section_name,
                    headings=tuple(current_headings),
                    body=body,
                )
            )

        current_body.clear()

    for line in markdown.splitlines():
        heading_match = MARKDOWN_HEADING_PATTERN.match(line)

        if heading_match is None:
            current_body.append(line)
            continue

        heading_level = len(heading_match.group(1))
        heading_text = _clean_heading(heading_match.group(2))

        if not heading_text:
            continue

        if heading_level == 1:
            flush_section()

            document_title = heading_text
            current_section_name = "Overview"
            current_headings = []
            current_level_two_heading = None
            continue

        if heading_level == 2:
            flush_section()

            current_level_two_heading = heading_text
            current_section_name = heading_text
            current_headings = [
                f"## {heading_text}",
            ]
            continue

        flush_section()

        current_section_name = heading_text
        current_headings = []

        if current_level_two_heading:
            current_headings.append(f"## {current_level_two_heading}")

        current_headings.append(f"### {heading_text}")

    flush_section()

    return document_title, sections


def _split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text near natural boundaries with character overlap."""

    normalized_text = text.strip()

    if not normalized_text:
        return []

    if len(normalized_text) <= chunk_size:
        return [normalized_text]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        maximum_end = min(
            start + chunk_size,
            text_length,
        )

        if maximum_end == text_length:
            end = text_length
        else:
            end = _find_preferred_boundary(
                normalized_text,
                start=start,
                maximum_end=maximum_end,
            )

        if end <= start:
            end = maximum_end

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(
            start + 1,
            end - chunk_overlap,
        )

        next_start = _move_to_word_boundary(
            normalized_text,
            start=next_start,
            maximum=end,
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def _find_preferred_boundary(
    text: str,
    *,
    start: int,
    maximum_end: int,
) -> int:
    """Find a natural split point before the maximum size."""

    minimum_boundary = start + int((maximum_end - start) * 0.60)

    separators = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
    )

    for separator in separators:
        position = text.rfind(
            separator,
            minimum_boundary,
            maximum_end,
        )

        if position != -1:
            return position + len(separator)

    whitespace_position = text.rfind(
        " ",
        start + 1,
        maximum_end,
    )

    if whitespace_position != -1:
        return whitespace_position + 1

    return maximum_end


def _move_to_word_boundary(
    text: str,
    *,
    start: int,
    maximum: int,
) -> int:
    """Move the start forward so a chunk does not begin mid-word."""

    position = start

    while position < maximum and not text[position].isspace():
        position += 1

    while position < len(text) and text[position].isspace():
        position += 1

    return position


def _validate_chunk_configuration(
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Validate character-based chunk settings."""

    if chunk_size <= 0:
        raise ValueError("RAG chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("RAG chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("RAG chunk overlap must be smaller than chunk size.")


def _normalize_markdown(markdown: str) -> str:
    """Normalize line endings and trailing whitespace."""

    normalized_lines = [
        line.rstrip()
        for line in markdown.replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .splitlines()
    ]

    return "\n".join(normalized_lines).strip()


def _clean_heading(heading: str) -> str:
    """Remove optional closing Markdown heading markers."""

    return re.sub(
        r"\s+#+\s*$",
        "",
        heading,
    ).strip()


def _slugify(value: str) -> str:
    """Create a stable ASCII identifier component."""

    normalized = unicodedata.normalize(
        "NFKD",
        value,
    )

    ascii_value = normalized.encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        ascii_value,
    )

    return slug.strip("-").lower()


def _create_content_hash(content: str) -> str:
    """Create a deterministic hash used to remove duplicates."""

    normalized_content = " ".join(content.split())

    return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
