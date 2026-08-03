"""Application configuration and project paths."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "xgboost_model.pkl"
MODELING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
DAILY_SALES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_product_sales.csv"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "motoretail.csv"
DATABASE_PATH = PROJECT_ROOT / "storage" / "motostock.db"

SELECTED_MODEL_NAME = "xgboost"
DEFAULT_LEAD_TIME_DAYS = 7

VALID_STOCK_STATUSES = ("critical", "warning", "healthy", "overstock")


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings shared by application services.

    The project intentionally keeps configuration small and dependency-free.
    Relative paths are resolved from the repository root so commands behave
    consistently when launched from different working directories.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 120.0
    rag_knowledge_base_path: Path = KNOWLEDGE_BASE_DIR
    rag_storage_path: Path = PROJECT_ROOT / "storage" / "rag"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_default_top_k: int = 4
    rag_max_top_k: int = 10
    rag_min_score: float | None = 0.40
    rag_max_context_chars: int = 6000
    data_backend: str = "sqlite"
    database_path: Path = DATABASE_PATH
    auto_import_csv: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Build validated settings from environment variables."""

        return cls(
            ollama_base_url=_read_text_setting(
                "OLLAMA_BASE_URL",
                default=cls.ollama_base_url,
            ).rstrip("/"),
            ollama_model=_read_text_setting(
                "OLLAMA_MODEL",
                default=cls.ollama_model,
            ),
            ollama_embedding_model=_read_text_setting(
                "OLLAMA_EMBEDDING_MODEL",
                default=cls.ollama_embedding_model,
            ),
            ollama_timeout_seconds=_read_float_setting(
                "OLLAMA_TIMEOUT_SECONDS",
                default=cls.ollama_timeout_seconds,
            ),
            rag_knowledge_base_path=_read_path_setting(
                "RAG_KNOWLEDGE_BASE_PATH",
                default=cls.rag_knowledge_base_path,
            ),
            rag_storage_path=_read_path_setting(
                "RAG_STORAGE_PATH",
                default=cls.rag_storage_path,
            ),
            rag_chunk_size=_read_int_setting(
                "RAG_CHUNK_SIZE",
                default=cls.rag_chunk_size,
            ),
            rag_chunk_overlap=_read_int_setting(
                "RAG_CHUNK_OVERLAP",
                default=cls.rag_chunk_overlap,
            ),
            rag_default_top_k=_read_int_setting(
                "RAG_DEFAULT_TOP_K",
                default=cls.rag_default_top_k,
            ),
            rag_max_top_k=_read_int_setting(
                "RAG_MAX_TOP_K",
                default=cls.rag_max_top_k,
            ),
            rag_min_score=_read_optional_float_setting(
                "RAG_MIN_SCORE",
                default=cls.rag_min_score,
            ),
            rag_max_context_chars=_read_int_setting(
                "RAG_MAX_CONTEXT_CHARS",
                default=cls.rag_max_context_chars,
            ),
            data_backend=_read_text_setting(
                "DATA_BACKEND",
                default=cls.data_backend,
            ).lower(),
            database_path=_read_path_setting(
                "DATABASE_PATH",
                default=cls.database_path,
            ),
            auto_import_csv=_read_bool_setting(
                "AUTO_IMPORT_CSV",
                default=cls.auto_import_csv,
            ),
        )

    def __post_init__(self) -> None:
        """Reject invalid values before a service starts using them."""

        if not self.ollama_base_url.strip():
            raise ValueError("OLLAMA_BASE_URL cannot be empty.")

        if not self.ollama_model.strip():
            raise ValueError("OLLAMA_MODEL cannot be empty.")

        if not self.ollama_embedding_model.strip():
            raise ValueError("OLLAMA_EMBEDDING_MODEL cannot be empty.")

        if not math.isfinite(self.ollama_timeout_seconds) or (
            self.ollama_timeout_seconds <= 0
        ):
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero.")

        if self.rag_chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than zero.")

        if self.rag_chunk_overlap < 0:
            raise ValueError("RAG_CHUNK_OVERLAP cannot be negative.")

        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

        if self.rag_default_top_k <= 0:
            raise ValueError("RAG_DEFAULT_TOP_K must be greater than zero.")

        if self.rag_max_top_k <= 0:
            raise ValueError("RAG_MAX_TOP_K must be greater than zero.")

        if self.rag_default_top_k > self.rag_max_top_k:
            raise ValueError("RAG_DEFAULT_TOP_K cannot exceed RAG_MAX_TOP_K.")

        if self.rag_min_score is not None and not 0 <= self.rag_min_score <= 1:
            raise ValueError("RAG_MIN_SCORE must be between zero and one.")

        if self.rag_max_context_chars <= 0:
            raise ValueError("RAG_MAX_CONTEXT_CHARS must be greater than zero.")

        if self.data_backend not in {"csv", "sqlite"}:
            raise ValueError("DATA_BACKEND must be either 'csv' or 'sqlite'.")

        if not str(self.database_path).strip():
            raise ValueError("DATABASE_PATH cannot be empty.")


def load_settings() -> Settings:
    """Load a fresh settings object from the current environment."""

    return Settings.from_env()


def _read_text_setting(name: str, *, default: str) -> str:
    """Read a non-empty text setting."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    value = raw_value.strip()

    if not value:
        raise ValueError(f"{name} cannot be empty.")

    return value


def _read_int_setting(name: str, *, default: int) -> int:
    """Read an integer setting with a useful configuration error."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer.") from exc


def _read_float_setting(name: str, *, default: float) -> float:
    """Read a floating-point setting with a useful configuration error."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number.") from exc


def _read_bool_setting(name: str, *, default: bool) -> bool:
    """Read a boolean setting using explicit, human-readable values."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean value.")


def _read_optional_float_setting(
    name: str,
    *,
    default: float | None = None,
) -> float | None:
    """Read an optional floating-point setting."""

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number.") from exc


def _read_path_setting(name: str, *, default: Path) -> Path:
    """Resolve a path setting relative to the repository when necessary."""

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    path = Path(raw_value.strip())

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()
