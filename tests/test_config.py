"""Tests for centralized runtime configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import PROJECT_ROOT, Settings, load_settings


def test_settings_resolve_relative_rag_paths_from_project_root(monkeypatch) -> None:
    monkeypatch.setenv("RAG_KNOWLEDGE_BASE_PATH", "custom-knowledge")
    monkeypatch.setenv("RAG_STORAGE_PATH", "custom-storage/rag")
    monkeypatch.setenv("RAG_DEFAULT_TOP_K", "3")
    monkeypatch.setenv("RAG_MAX_TOP_K", "7")

    settings = load_settings()

    assert settings.rag_knowledge_base_path == (
        PROJECT_ROOT / "custom-knowledge"
    ).resolve()
    assert settings.rag_storage_path == (
        PROJECT_ROOT / "custom-storage" / "rag"
    ).resolve()
    assert settings.rag_default_top_k == 3
    assert settings.rag_max_top_k == 7


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("RAG_DEFAULT_TOP_K", "0", "RAG_DEFAULT_TOP_K"),
        ("RAG_MAX_TOP_K", "0", "RAG_MAX_TOP_K"),
        ("RAG_CHUNK_SIZE", "invalid", "RAG_CHUNK_SIZE"),
        ("RAG_CHUNK_OVERLAP", "1000", "RAG_CHUNK_OVERLAP"),
        ("RAG_MIN_SCORE", "1.5", "RAG_MIN_SCORE"),
    ],
)
def test_invalid_settings_raise_clear_errors(monkeypatch, name, value, message) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        load_settings()


def test_settings_accept_explicit_paths_and_optional_threshold(tmp_path: Path) -> None:
    settings = Settings(
        rag_knowledge_base_path=tmp_path / "knowledge",
        rag_storage_path=tmp_path / "storage",
        rag_min_score=0.4,
    )

    assert settings.rag_knowledge_base_path == tmp_path / "knowledge"
    assert settings.rag_storage_path == tmp_path / "storage"
    assert settings.rag_min_score == 0.4


def test_settings_parse_sqlite_backend_and_boolean(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_BACKEND", "SQLITE")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "motostock.db"))
    monkeypatch.setenv("AUTO_IMPORT_CSV", "false")

    settings = load_settings()

    assert settings.data_backend == "sqlite"
    assert settings.database_path == (tmp_path / "motostock.db").resolve()
    assert settings.auto_import_csv is False


def test_invalid_backend_and_boolean_raise_clear_errors(monkeypatch):
    for name, value in [("DATA_BACKEND", "postgres"), ("AUTO_IMPORT_CSV", "maybe")]:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            load_settings()
        monkeypatch.delenv(name)
