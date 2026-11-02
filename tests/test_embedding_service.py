"""Tests for the embedding service."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pdf_assistant.models.schemas import TextChunk
from pdf_assistant.services import embedding_service
from pdf_assistant.services.embedding_service import (
    EmbeddingError,
    embed_chunks,
    embed_query,
    embed_texts,
    load_model,
)


class FakeEmbedder:
    """A deterministic fake that replaces SentenceTransformer in tests.

    Produces a fixed 4-dimensional embedding for each text without downloading
    any model or touching the network. Tests inject this in place of the real
    model so they can run fast and offline.
    """

    def __init__(self):
        """Track which texts were encoded, for tests to inspect."""
        self.encoded_texts: list[str] = []

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Return a deterministic (len(texts), 4) array.

        Each row is [1.0, 2.0, 3.0, row_index], so different texts get
        different embeddings and tests can distinguish them.
        """
        self.encoded_texts.extend(texts)
        embeddings = []
        for i in range(len(texts)):
            embeddings.append([1.0, 2.0, 3.0, float(i)])
        return np.array(embeddings, dtype=np.float32)


def test_embed_texts_returns_one_row_per_text():
    """Each text becomes exactly one row in the output."""
    fake = FakeEmbedder()
    texts = ["first", "second", "third"]

    result = embed_texts(fake, texts)

    assert result.shape == (3, 4)


def test_embed_texts_returns_float32():
    """FAISS requires float32, so embeddings must use that dtype."""
    fake = FakeEmbedder()
    texts = ["one", "two"]

    result = embed_texts(fake, texts)

    assert result.dtype == np.float32


def test_embed_texts_rows_are_l2_normalised():
    """Every embedding must have unit length for cosine similarity to work.

    Task 2 (vector search) relies on normalised embeddings so inner product
    equals cosine similarity. This guarantee must hold even when the model or
    fake does not normalise its output.
    """
    fake = FakeEmbedder()
    texts = ["alpha", "beta", "gamma"]

    result = embed_texts(fake, texts)

    for row in result:
        norm = np.linalg.norm(row)
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_embed_query_returns_a_single_row():
    """A query is one question, so one row with shape (1, dim)."""
    fake = FakeEmbedder()
    question = "What is the document about?"

    result = embed_query(fake, question)

    assert result.shape == (1, 4)
    # Also check it is normalised
    norm = np.linalg.norm(result[0])
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_embed_chunks_uses_chunk_text():
    """embed_chunks extracts .text from each chunk and passes it to embed_texts."""
    fake = FakeEmbedder()
    chunks = [
        TextChunk(chunk_id=0, text="First chunk text", page_number=1),
        TextChunk(chunk_id=1, text="Second chunk text", page_number=2),
    ]

    result = embed_chunks(fake, chunks)

    # The fake records what it saw
    assert fake.encoded_texts == ["First chunk text", "Second chunk text"]
    # And we get back the right shape
    assert result.shape == (2, 4)


def test_embed_texts_rejects_an_empty_list():
    """Embedding zero texts is a programming error, not valid input."""
    fake = FakeEmbedder()

    with pytest.raises(EmbeddingError, match="at least one text"):
        embed_texts(fake, [])


def test_embed_query_rejects_blank_question():
    """Empty or whitespace-only questions must be caught before embedding."""
    fake = FakeEmbedder()

    with pytest.raises(EmbeddingError, match="blank"):
        embed_query(fake, "")

    with pytest.raises(EmbeddingError, match="blank"):
        embed_query(fake, "   ")


def test_load_model_is_cached():
    """The model must be loaded once per process, not on every call.

    Streamlit re-runs the entire script on every interaction, so without
    caching the app would reload the 90 MB model and waste several seconds
    every time the user types a message. functools.lru_cache(maxsize=1) solves
    this by keeping the last-loaded model in memory.
    """
    # Replace the real SentenceTransformer with a counting stub
    call_count = 0

    def fake_constructor(model_name: str):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        mock.model_name = model_name
        return mock

    original = embedding_service.SentenceTransformer
    embedding_service.SentenceTransformer = fake_constructor

    try:
        # Clear any prior cached value
        load_model.cache_clear()

        # Load the same model twice
        model_a = load_model("test-model")
        model_b = load_model("test-model")

        # Should construct exactly once
        assert call_count == 1
        # Should return the same object
        assert model_a is model_b

        # Clear the cache again so we leave the module clean
        load_model.cache_clear()
    finally:
        # Restore the real class
        embedding_service.SentenceTransformer = original
