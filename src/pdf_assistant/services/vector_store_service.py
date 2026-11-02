"""Build and search a FAISS vector index for semantic similarity retrieval.

This is the third step of the RAG pipeline. Phase 2 extracted and chunked the
PDF; Task 1 (embedding_service) turned those chunks into vectors. Now those
vectors go into a searchable index so the user's question can retrieve the most
relevant chunks.

WHY INDEXFLATIP: FAISS offers dozens of index types. We use IndexFlatIP (exact
brute-force inner-product search) for three reasons:

1. No training required. Approximate indexes (IVF, HNSW, etc.) need a training
   step that a BCA student would have to explain in a viva.
2. Exact results. Every search returns the true nearest neighbours. For a
   student project, reproducibility and ease of debugging beat speed.
3. Student-sized documents. A BCA final-year project will test this on PDFs with
   at most a few hundred chunks. Brute force over 500 vectors takes microseconds;
   the complexity of an approximate index would be pure overhead.

WHY INNER PRODUCT EQUALS COSINE SIMILARITY: embedding_service.py normalises
every vector to unit length (L2 norm = 1). For unit vectors, the inner product
of two vectors equals their cosine similarity, because the denominator in the
cosine formula is always 1. FAISS can compute inner products extremely fast,
so by normalising first, we get cosine similarity at inner-product speed.
"""

from __future__ import annotations

import faiss
import numpy as np


class VectorStoreError(RuntimeError):
    """Raised when a vector index cannot be built or searched."""


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS index from a batch of normalised embedding vectors.

    Args:
        vectors: A 2D float32 array of shape ``(num_vectors, embedding_dim)``
            where every row is L2-normalised to unit length. This comes from
            ``embedding_service.embed_texts`` or ``embed_chunks``.

    Returns:
        A ``faiss.IndexFlatIP`` ready to search. The index holds all vectors
        and supports exact nearest-neighbour queries via inner product.

    Raises:
        VectorStoreError: If the array is empty, not 2D, or invalid.
    """
    # Validate shape before FAISS sees it - FAISS error messages are cryptic.
    if vectors.ndim != 2:
        raise VectorStoreError(
            f"build_index requires a two-dimensional array of shape "
            f"(num_vectors, embedding_dim), but received shape {vectors.shape}"
        )

    if vectors.shape[0] == 0:
        raise VectorStoreError(
            "build_index requires at least one vector, but received an empty array"
        )

    embedding_dim = vectors.shape[1]

    try:
        # FAISS requires contiguous float32. The embedding service already
        # produces float32, but np.ascontiguousarray is cheap when the input is
        # already contiguous and prevents obscure failures when it is not.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        # IndexFlatIP = exact brute-force search via inner product.
        # See module docstring for why this is the right choice.
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(vectors)

        return index

    except Exception as exc:
        raise VectorStoreError(
            f"Failed to build FAISS index from {vectors.shape[0]} vectors "
            f"of dimension {embedding_dim}. Technical detail: {exc}"
        ) from exc


def search(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Search the index for the nearest neighbours of a query vector.

    Args:
        index: A FAISS index built by ``build_index``.
        query_vector: A normalised embedding of shape ``(1, embedding_dim)``,
            typically from ``embedding_service.embed_query``.
        top_k: How many nearest neighbours to return. Must be at least 1.

    Returns:
        A list of ``(row_index, similarity_score)`` tuples, ordered from most
        similar to least similar. Length is ``min(top_k, index.ntotal)`` - if
        the index holds fewer vectors than requested, only the actual matches
        are returned (FAISS pads with ``-1`` indices, which this function
        filters out before returning).

        ``row_index`` is the position of the chunk in the original list passed
        to ``build_index``. The caller uses this to look up the corresponding
        :class:`TextChunk` and extract its page number for citation.

        ``similarity_score`` is the inner product of the query and the matched
        vector. Because both are normalised, this equals cosine similarity and
        ranges from -1 (opposite direction) to +1 (identical direction). In
        practice, semantic search scores are usually between 0.0 and 1.0.

    Raises:
        VectorStoreError: If ``top_k`` is below 1, the query vector is not 2D,
            or the query vector's dimension does not match the index.
    """
    if top_k < 1:
        raise VectorStoreError(f"top_k must be at least 1, got {top_k}")

    # Validate the shape BEFORE indexing into it. A 1-D array of shape (dim,)
    # has no shape[1], so reading it would raise a raw IndexError out of a
    # service that is supposed to speak only VectorStoreError. embed_query
    # always returns a 2-D array, but search is public and a future caller
    # could reasonably pass the flat vector by mistake.
    if query_vector.ndim != 2:
        raise VectorStoreError(
            f"search requires a two-dimensional query array of shape "
            f"(1, embedding_dim), but received shape {query_vector.shape}. "
            f"Reshape a flat vector with .reshape(1, -1)."
        )

    # Validate dimension match before FAISS sees it
    if query_vector.shape[1] != index.d:
        raise VectorStoreError(
            f"Query vector dimension {query_vector.shape[1]} does not match "
            f"index dimension {index.d}"
        )

    try:
        # FAISS requires contiguous float32
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # search() returns (distances, indices) - both are 2D arrays of shape
        # (num_queries, top_k). We pass one query, so we extract row 0.
        scores, indices = index.search(query_vector, top_k)
        scores = scores[0]  # shape: (top_k,)
        indices = indices[0]  # shape: (top_k,)

        # CRITICAL: FAISS pads with -1 when it has fewer vectors than top_k.
        # For example, if the index holds 2 vectors and top_k=4, indices will
        # be [1, 0, -1, -1] with meaningless scores for the last two positions.
        # A -1 leaking out would index a Python list from the end, silently
        # citing the wrong page. Filter them out here.
        results = [
            (int(idx), float(score)) for idx, score in zip(indices, scores, strict=True) if idx >= 0
        ]

        return results

    except VectorStoreError:
        # Already a clear message - pass through
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to search index with query of dimension "
            f"{query_vector.shape[1]}. Technical detail: {exc}"
        ) from exc
