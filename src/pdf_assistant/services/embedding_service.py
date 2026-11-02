"""Turn text into vectors that the retrieval step can search.

Embedding is the second step of the RAG pipeline. The PDF was extracted and
chunked in Phase 2; now each chunk becomes a fixed-length vector. Task 2 (the
vector index) will use these embeddings to find the chunks most similar to the
user's question.

WHY NORMALISATION MATTERS: cosine similarity is what makes semantic search work,
because it measures the *direction* of a vector, ignoring its magnitude. Two
chunks about the same topic point in the same direction even if one is longer.
Cosine similarity is expensive to compute, but there is a trick: if every vector
has unit length (L2 norm = 1), then cosine similarity equals inner product,
which FAISS can compute very fast. That is why every embedding produced here is
normalised explicitly, regardless of what the model does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_assistant.models.schemas import TextChunk


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded."""


@lru_cache(maxsize=1)
def load_model(model_name: str) -> SentenceTransformer:
    """Load the embedding model, caching it in memory for subsequent calls.

    Streamlit re-runs the entire script on every interaction. Without caching,
    the app would reload the 90 MB model and waste several seconds every time
    the user typed a message. ``functools.lru_cache(maxsize=1)`` solves this by
    keeping the last-loaded model in memory across re-runs.

    Args:
        model_name: Hugging Face model identifier, e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``.

    Returns:
        A loaded SentenceTransformer ready to encode text.

    Raises:
        EmbeddingError: If the model cannot be loaded (network failure, invalid
            name, corrupted cache).
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load the embedding model '{model_name}'. "
            f"Check your internet connection and verify the model name. "
            f"Technical detail: {exc}"
        ) from exc


def embed_texts(model: Any, texts: list[str]) -> np.ndarray:
    """Turn a batch of texts into normalised embeddings.

    Args:
        model: An object with an ``.encode(texts, **kwargs)`` method. In
            production this is a SentenceTransformer; in tests it is a fake.
        texts: Non-empty list of strings to embed.

    Returns:
        A float32 array of shape ``(len(texts), embedding_dim)`` where every row
        is L2-normalised to unit length. Normalisation is done explicitly here
        rather than relying on the model, so the guarantee holds even for
        injected fakes in tests.

    Raises:
        EmbeddingError: If the list is empty or the model fails.
    """
    if not texts:
        raise EmbeddingError("embed_texts requires at least one text")

    try:
        # Convert to float32 because FAISS requires it. Do NOT rely on
        # encode(normalize_embeddings=True) because injected test fakes will not
        # have that parameter, and we want the normalisation guarantee to be
        # visible in our code, not hidden inside the model.
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Explicitly normalise each row to unit length.
        # WHY: makes inner product equal cosine similarity (see module docstring).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero if the model returns a zero vector
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return embeddings

    except EmbeddingError:
        # Already a clear message - pass through
        raise
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to compute embeddings for {len(texts)} texts. Technical detail: {exc}"
        ) from exc


def embed_chunks(model: Any, chunks: list[TextChunk]) -> np.ndarray:
    """Convenience wrapper that embeds the text of every chunk.

    Args:
        model: An object with an ``.encode`` method.
        chunks: Non-empty list of :class:`TextChunk` objects.

    Returns:
        A float32 array of shape ``(len(chunks), embedding_dim)`` with every row
        normalised.

    Raises:
        EmbeddingError: If the list is empty or embedding fails.
    """
    texts = [chunk.text for chunk in chunks]
    return embed_texts(model, texts)


def embed_query(model: Any, question: str) -> np.ndarray:
    """Embed a single user question.

    Args:
        model: An object with an ``.encode`` method.
        question: The user's question.

    Returns:
        A float32 array of shape ``(1, embedding_dim)``, normalised.

    Raises:
        EmbeddingError: If the question is blank or embedding fails.
    """
    if not question.strip():
        raise EmbeddingError("Cannot embed a blank question")

    return embed_texts(model, [question])
