"""Application configuration.

All settings come from the local ``.env`` file (see ``.env.example``). This is
the ONLY module that reads environment variables, so there is exactly one place
to look when a setting needs to change.

The Groq API key is never printed, logged or shown in the user interface. The
rest of the application only ever asks "is a key present?" through
:meth:`Settings.has_api_key`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Project root = three levels up from this file:
#   src/pdf_assistant/config.py -> src/pdf_assistant -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Defaults used when a value is missing from .env. Keeping them here means the
# application still starts with sensible behaviour on a fresh machine.
# Verified against the Groq models endpoint on 2026-11-01. Groq retires models
# regularly, so if this ever fails with "model not found", check the current
# list at https://console.groq.com/docs/models and set GROQ_MODEL in .env.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_MAX_UPLOAD_SIZE_MB = 20
DEFAULT_MIN_SIMILARITY = 0.25


class ConfigError(ValueError):
    """Raised when a configuration value is present but not usable."""


def _read_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer setting from the environment.

    Falls back to ``default`` when the variable is unset or blank. A value that
    is set but not a valid integer is treated as a configuration mistake rather
    than silently ignored, so the student sees the problem immediately.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, but got {raw!r}.") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, but got {value}.")
    return value


def _read_float(name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Read a float setting from the environment.

    Falls back to ``default`` when the variable is unset or blank. A value that
    is set but not a valid float is treated as a configuration mistake rather
    than silently ignored, so the student sees the problem immediately.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, but got {raw!r}.") from exc
    if value < minimum or value > maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, but got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    """Every setting the application needs, resolved once at startup."""

    groq_api_key: str
    groq_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_upload_size_mb: int
    min_similarity: float

    def __repr__(self) -> str:
        """Render the settings with the API key masked.

        A frozen dataclass gets a free ``__repr__`` that prints every field,
        including ``groq_api_key``. That makes one careless ``st.write(settings)``
        or f-string in the UI enough to put the secret on screen - and Phase 4
        extends the configuration table in ``ui/document_details.py``, which is
        exactly where that would happen. Masking the key here means no caller can
        leak it by accident, however it is printed. ``__str__`` falls back to
        ``__repr__``, so both are covered.
        """
        key_state = "***set***" if self.has_api_key else "***missing***"
        return (
            f"Settings(groq_api_key={key_state}, "
            f"groq_model={self.groq_model!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"top_k={self.top_k}, "
            f"max_upload_size_mb={self.max_upload_size_mb}, "
            f"min_similarity={self.min_similarity})"
        )

    @property
    def has_api_key(self) -> bool:
        """True when a Groq API key is configured.

        The UI uses this to disable AI features and show a helpful message
        instead of crashing when the key is missing.
        """
        return bool(self.groq_api_key.strip())

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload limit in bytes, which is what file sizes are measured in."""
        return self.max_upload_size_mb * 1024 * 1024

    def validate(self) -> None:
        """Check settings that depend on each other.

        Chunk overlap must be smaller than chunk size. If the overlap were equal
        to or larger than the chunk size, the chunker would never move forward
        and would loop forever producing the same text.

        Min similarity must be in [0.0, 1.0] because it is a cosine similarity
        threshold. This check is redundant with _read_float's validation, but
        kept here for consistency with the chunk_overlap check and to guard
        against direct Settings construction in tests.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size}), otherwise chunking cannot make progress."
            )
        if not (0.0 <= self.min_similarity <= 1.0):
            raise ConfigError(
                f"MIN_SIMILARITY must be between 0.0 and 1.0, but got {self.min_similarity}."
            )


def load_settings(env_file: Path | None = None) -> Settings:
    """Build a :class:`Settings` object from the ``.env`` file and environment.

    Args:
        env_file: Optional path to an env file. Defaults to ``.env`` in the
            project root. Values already set in the real environment win, so
            tests can override them with ``monkeypatch``.
    """
    load_dotenv(dotenv_path=env_file or (PROJECT_ROOT / ".env"), override=False)

    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL,
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL,
        chunk_size=_read_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE, minimum=100),
        chunk_overlap=_read_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, minimum=0),
        top_k=_read_int("TOP_K", DEFAULT_TOP_K, minimum=1),
        max_upload_size_mb=_read_int("MAX_UPLOAD_SIZE_MB", DEFAULT_MAX_UPLOAD_SIZE_MB, minimum=1),
        min_similarity=_read_float(
            "MIN_SIMILARITY", DEFAULT_MIN_SIMILARITY, minimum=0.0, maximum=1.0
        ),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared settings object.

    Cached because Streamlit re-runs the whole script on every interaction, and
    there is no reason to re-read the .env file each time.
    """
    return load_settings()
