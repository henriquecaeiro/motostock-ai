"""Validation for the versioned assistant evaluation set."""

from __future__ import annotations

import json
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"


def test_evaluation_files_are_valid_and_cross_referenced() -> None:
    rag = json.loads((EVALUATION_DIR / "rag_questions.json").read_text(encoding="utf-8"))
    tools = json.loads((EVALUATION_DIR / "tool_questions.json").read_text(encoding="utf-8"))
    safety = json.loads((EVALUATION_DIR / "safety_questions.json").read_text(encoding="utf-8"))
    expected_sources = json.loads(
        (EVALUATION_DIR / "expected_sources.json").read_text(encoding="utf-8")
    )

    assert rag and tools and safety
    assert {item["id"] for item in rag} <= set(expected_sources)
    assert all("question" in item and "expected_tool" in item for item in tools)
    assert all("question" in item and "expected_behavior" in item for item in safety)
