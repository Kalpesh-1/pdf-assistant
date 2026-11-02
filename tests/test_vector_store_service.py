"""Tests for the vector store service (FAISS index operations)."""

from __future__ import annotations

import numpy as np
import pytest

from pdf_assistant.services.vector_store_service import (
    VectorStoreError,
    build_index,
    search,
)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors to unit length, matching embedding_service behaviour."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def test_build_index_contains_every_vector():
    """Index should hold all vectors added to it."""
    vectors = normalize_vectors(
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )
    index = build_index(vectors)
    assert index.ntotal == 3


def test_build_index_uses_the_vector_width():
    """Index dimension should match the vectors' embedding dimension."""
    vectors = normalize_vectors(
        np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
    )
    index = build_index(vectors)
    assert index.d == 4


def test_search_returns_the_nearest_vector_first():
    """Querying with a copy of a vector should return that vector first."""
    vectors = normalize_vectors(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    index = build_index(vectors)
    # Query with a copy of row 2 (third vector)
    query = vectors[2:3]
    results = search(index, query, top_k=3)
    # Should return index 2 first
    assert results[0][0] == 2


def test_search_scores_are_descending():
    """Search results should be ordered from highest to lowest similarity."""
    vectors = normalize_vectors(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    index = build_index(vectors)
    query = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    results = search(index, query, top_k=4)
    scores = [score for _, score in results]
    # Check that scores are in descending order
    assert scores == sorted(scores, reverse=True)


def test_search_identical_vector_scores_about_one():
    """An exact match should score very close to 1.0 (perfect cosine similarity)."""
    vectors = normalize_vectors(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    )
    index = build_index(vectors)
    # Query with exact copy of the first vector
    query = vectors[0:1]
    results = search(index, query, top_k=1)
    _, score = results[0]
    assert score == pytest.approx(1.0, abs=1e-5)


def test_search_returns_top_k_results():
    """When top_k is less than the index size, return exactly top_k results."""
    vectors = normalize_vectors(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    index = build_index(vectors)
    query = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    results = search(index, query, top_k=3)
    assert len(results) == 3


def test_search_caps_at_index_size_when_top_k_is_larger():
    """When top_k exceeds index size, return only actual matches (no -1 padding)."""
    vectors = normalize_vectors(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    )
    index = build_index(vectors)
    query = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    # Ask for 4 results but only 2 vectors exist
    results = search(index, query, top_k=4)
    # Should return exactly 2, not 4 with -1 padding
    assert len(results) == 2
    # Verify no -1 indices leaked through
    for idx, _ in results:
        assert idx >= 0


def test_build_index_rejects_an_empty_array():
    """Building an index from zero vectors should fail."""
    empty = np.array([], dtype=np.float32).reshape(0, 4)
    with pytest.raises(VectorStoreError, match="at least one vector"):
        build_index(empty)


def test_build_index_rejects_a_one_dimensional_array():
    """Vectors must be 2D (rows of embeddings), not a flat 1D array."""
    flat = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with pytest.raises(VectorStoreError, match="two-dimensional"):
        build_index(flat)


def test_search_rejects_top_k_below_one():
    """top_k must be at least 1."""
    vectors = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    index = build_index(vectors)
    query = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    with pytest.raises(VectorStoreError, match="at least 1"):
        search(index, query, top_k=0)


def test_search_rejects_a_width_mismatch():
    """Query vector must have the same dimension as the index."""
    vectors = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    index = build_index(vectors)
    # Query has wrong dimension (3 instead of 4)
    wrong_query = normalize_vectors(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    with pytest.raises(VectorStoreError, match="dimension"):
        search(index, wrong_query, top_k=1)


def test_search_rejects_a_one_dimensional_query():
    """A flat query vector must raise VectorStoreError, not a raw IndexError.

    embed_query always returns shape (1, dim), so this is unreachable through the
    normal pipeline - but search is public, and reading shape[1] on a 1-D array
    used to leak a bare IndexError out of a service that is supposed to speak
    only VectorStoreError.
    """
    vectors = normalize_vectors(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    index = build_index(vectors)
    flat_query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(VectorStoreError, match="two-dimensional"):
        search(index, flat_query, top_k=1)
