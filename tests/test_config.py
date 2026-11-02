"""Tests for configuration loading."""

from __future__ import annotations

import pytest

from pdf_assistant.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MAX_UPLOAD_SIZE_MB,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    ConfigError,
    Settings,
    load_settings,
)

# Every environment variable the application reads. Tests clear all of them so
# that a real .env file on the developer's machine cannot change the results.
ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "MAX_UPLOAD_SIZE_MB",
    "MIN_SIMILARITY",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Remove all application env vars and point .env at an empty file."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    return empty_env


def test_defaults_match_the_specification():
    """Pin the defaults to the values the project specification lists.

    Comparing settings against the module's own constants would pass even if a
    constant drifted, so the constants themselves are checked here.
    """
    assert DEFAULT_CHUNK_SIZE == 800
    assert DEFAULT_CHUNK_OVERLAP == 150
    assert DEFAULT_TOP_K == 4
    assert DEFAULT_MAX_UPLOAD_SIZE_MB == 20
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_defaults_are_used_when_nothing_is_set(clean_env):
    settings = load_settings(env_file=clean_env)

    assert settings.chunk_size == DEFAULT_CHUNK_SIZE
    assert settings.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert settings.top_k == DEFAULT_TOP_K
    assert settings.max_upload_size_mb == DEFAULT_MAX_UPLOAD_SIZE_MB
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.groq_model == DEFAULT_GROQ_MODEL


def test_env_values_override_defaults(clean_env, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")

    settings = load_settings(env_file=clean_env)

    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200
    assert settings.top_k == 6
    assert settings.groq_model == "some-other-model"


def test_missing_api_key_is_reported_not_crashed(clean_env):
    """A missing key must be a reportable state, not an exception.

    The application needs to start without a key so it can show a helpful
    message in the UI.
    """
    settings = load_settings(env_file=clean_env)

    assert settings.groq_api_key == ""
    assert settings.has_api_key is False


def test_present_api_key_is_detected(clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_key_for_testing")

    settings = load_settings(env_file=clean_env)

    assert settings.has_api_key is True


def test_whitespace_only_api_key_counts_as_missing(clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")

    assert load_settings(env_file=clean_env).has_api_key is False


def test_non_numeric_chunk_size_is_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "eight hundred")

    with pytest.raises(ConfigError, match="CHUNK_SIZE"):
        load_settings(env_file=clean_env)


def test_overlap_not_smaller_than_chunk_size_is_rejected(clean_env, monkeypatch):
    """Overlap >= chunk size would make the chunker loop forever."""
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        load_settings(env_file=clean_env)


def test_top_k_below_one_is_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ConfigError, match="TOP_K"):
        load_settings(env_file=clean_env)


def test_max_upload_size_converts_to_bytes():
    settings = Settings(
        groq_api_key="",
        groq_model="m",
        embedding_model="e",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )

    assert settings.max_upload_size_bytes == 20 * 1024 * 1024


def test_min_similarity_default_is_read(clean_env):
    """The default min_similarity matches DEFAULT_MIN_SIMILARITY."""
    assert DEFAULT_MIN_SIMILARITY == 0.25

    settings = load_settings(env_file=clean_env)
    assert settings.min_similarity == DEFAULT_MIN_SIMILARITY


def test_min_similarity_can_be_overridden(clean_env, monkeypatch):
    """MIN_SIMILARITY env var overrides the default."""
    monkeypatch.setenv("MIN_SIMILARITY", "0.5")

    settings = load_settings(env_file=clean_env)
    assert settings.min_similarity == 0.5


def test_non_numeric_min_similarity_is_rejected(clean_env, monkeypatch):
    """A non-numeric MIN_SIMILARITY raises ConfigError."""
    monkeypatch.setenv("MIN_SIMILARITY", "not a number")

    with pytest.raises(ConfigError, match="MIN_SIMILARITY"):
        load_settings(env_file=clean_env)


def test_min_similarity_outside_zero_to_one_is_rejected(clean_env, monkeypatch):
    """MIN_SIMILARITY must be in [0.0, 1.0]."""
    # Test above 1.0
    monkeypatch.setenv("MIN_SIMILARITY", "1.5")
    with pytest.raises(ConfigError, match="MIN_SIMILARITY"):
        load_settings(env_file=clean_env)

    # Test below 0.0
    monkeypatch.setenv("MIN_SIMILARITY", "-0.1")
    with pytest.raises(ConfigError, match="MIN_SIMILARITY"):
        load_settings(env_file=clean_env)


def test_repr_never_contains_the_api_key():
    """Printing Settings must never reveal the key.

    Settings is a frozen dataclass, so without an explicit __repr__ the free one
    prints every field including groq_api_key. One ``st.write(settings)`` in the
    UI would then put the secret on screen. ``str()`` falls back to ``__repr__``,
    so both are checked, and f-string interpolation is covered too.
    """
    secret = "gsk_ThisIsAFakeKeyShapedSecret1234567890"
    settings = Settings(
        groq_api_key=secret,
        groq_model="some-model",
        embedding_model="some-embedding-model",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in f"{settings}"
    assert secret not in f"{settings!r}"

    # The key's presence is still reported, and the harmless fields still show.
    assert "***set***" in repr(settings)
    assert "some-model" in repr(settings)
    assert "min_similarity=0.25" in repr(settings)


def test_repr_reports_a_missing_api_key_as_missing():
    """With no key configured, the repr says so rather than showing an empty string."""
    settings = Settings(
        groq_api_key="",
        groq_model="some-model",
        embedding_model="some-embedding-model",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )

    assert "***missing***" in repr(settings)
    assert "***set***" not in repr(settings)
