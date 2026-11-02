"""Tests for retrieval_service.py: the layer that joins embedding and search."""

from __future__ import annotations

import numpy as np
import pytest

from pdf_assistant.models.schemas import RetrievedChunk, TextChunk
from pdf_assistant.services import embedding_service, retrieval_service, vector_store_service
from pdf_assistant.services.embedding_service import EmbeddingError
from pdf_assistant.services.retrieval_service import RetrievalError


# Fake embedding model for tests. Returns the vector passed in as data.
class FakeEmbedder:
    """Test double that returns whatever vector you give it.

    Tests build the exact vectors they need with numpy, then inject this fake so
    retrieval_service gets predictable embeddings without touching the network or
    loading a real 90 MB model.
    """

    def __init__(self, vectors: dict[str, np.ndarray]):
        """Build a fake embedder.

        Args:
            vectors: A map from text to the vector that text should produce.
                Each vector must be shape (embedding_dim,) and will be normalised.
        """
        self.vectors = vectors

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Pretend to embed texts, actually returning pre-built vectors."""
        result = np.array([self.vectors[text] for text in texts], dtype=np.float32)
        return result


def test_retrieve_returns_the_matching_chunk_first():
    """The chunk most similar to the question ranks first."""
    chunks = [
        TextChunk(chunk_id=0, text="The sky is blue.", page_number=1),
        TextChunk(chunk_id=1, text="Python is a programming language.", page_number=2),
        TextChunk(chunk_id=2, text="Paris is the capital of France.", page_number=3),
    ]

    # Build vectors by hand: query matches chunk 1 most strongly.
    # All vectors are 4-dimensional for this test.
    query_vec = np.array([0.0, 1.0, 0.0, 0.0])
    chunk0_vec = np.array([1.0, 0.0, 0.0, 0.0])  # orthogonal to query
    chunk1_vec = np.array([0.0, 1.0, 0.0, 0.0])  # perfect match
    chunk2_vec = np.array([0.0, 0.5, 0.5, 0.0])  # partial overlap

    model = FakeEmbedder(
        {
            "What is Python?": query_vec,
            "The sky is blue.": chunk0_vec,
            "Python is a programming language.": chunk1_vec,
            "Paris is the capital of France.": chunk2_vec,
        }
    )

    # Embed chunks and build index
    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    # Retrieve
    results = retrieval_service.retrieve(
        question="What is Python?",
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
        min_similarity=0.0,
    )

    # chunk 1 should come first, with a perfect score of 1.0
    assert len(results) == 3
    assert results[0].chunk.chunk_id == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_retrieve_preserves_the_page_number_of_each_chunk():
    """The whole point of retrieval: citations must carry the real page number."""
    chunks = [
        TextChunk(chunk_id=0, text="First chunk.", page_number=5),
        TextChunk(chunk_id=1, text="Second chunk.", page_number=12),
    ]

    # Query matches chunk 0
    query_vec = np.array([1.0, 0.0])
    chunk0_vec = np.array([1.0, 0.0])
    chunk1_vec = np.array([0.0, 1.0])

    model = FakeEmbedder(
        {
            "question": query_vec,
            "First chunk.": chunk0_vec,
            "Second chunk.": chunk1_vec,
        }
    )

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    results = retrieval_service.retrieve(
        question="question",
        chunks=chunks,
        index=index,
        model=model,
        top_k=2,
        min_similarity=0.0,
    )

    # chunk 0 wins, and its page_number is preserved
    assert results[0].chunk.page_number == 5
    assert results[1].chunk.page_number == 12


def test_retrieve_returns_at_most_top_k():
    """When more chunks exist, only the best k are returned."""
    chunks = [TextChunk(chunk_id=i, text=f"Chunk {i}", page_number=i + 1) for i in range(10)]

    # All chunks get the same vector, so all are equally good matches
    uniform_vec = np.array([1.0, 0.0])
    model = FakeEmbedder({f"Chunk {i}": uniform_vec for i in range(10)} | {"q": uniform_vec})

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    results = retrieval_service.retrieve(
        question="q",
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
        min_similarity=0.0,
    )

    assert len(results) == 3


def test_retrieve_drops_results_below_the_floor():
    """Only chunks scoring >= min_similarity are returned."""
    chunks = [
        TextChunk(chunk_id=0, text="A", page_number=1),
        TextChunk(chunk_id=1, text="B", page_number=2),
        TextChunk(chunk_id=2, text="C", page_number=3),
    ]

    # chunk 0: perfect match (score 1.0)
    # chunk 1: partial match (score ~0.7)
    # chunk 2: weak match (score < 0.5)
    query_vec = np.array([1.0, 0.0, 0.0])
    chunk0_vec = np.array([1.0, 0.0, 0.0])
    chunk1_vec = np.array([0.7, 0.7, 0.0])  # normalised: ~(0.707, 0.707, 0)
    chunk2_vec = np.array([0.3, 0.95, 0.0])  # normalised: ~(0.302, 0.953, 0)

    model = FakeEmbedder(
        {
            "q": query_vec,
            "A": chunk0_vec,
            "B": chunk1_vec,
            "C": chunk2_vec,
        }
    )

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    # With min_similarity=0.9, only chunk 0 clears the floor
    results = retrieval_service.retrieve(
        question="q",
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
        min_similarity=0.9,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == 0


def test_retrieve_returns_empty_when_nothing_clears_the_floor():
    """When no chunk scores high enough, return [] rather than weak matches."""
    chunks = [
        TextChunk(chunk_id=0, text="Unrelated topic A", page_number=1),
        TextChunk(chunk_id=1, text="Unrelated topic B", page_number=2),
    ]

    # Query is orthogonal to both chunks (score = 0.0)
    query_vec = np.array([1.0, 0.0, 0.0])
    chunk0_vec = np.array([0.0, 1.0, 0.0])
    chunk1_vec = np.array([0.0, 0.0, 1.0])

    model = FakeEmbedder(
        {
            "q": query_vec,
            "Unrelated topic A": chunk0_vec,
            "Unrelated topic B": chunk1_vec,
        }
    )

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    results = retrieval_service.retrieve(
        question="q",
        chunks=chunks,
        index=index,
        model=model,
        top_k=2,
        min_similarity=0.5,
    )

    assert results == []


def test_has_sufficient_context_is_false_for_an_empty_list():
    """No chunks = no context = abstain."""
    assert retrieval_service.has_sufficient_context([]) is False


def test_has_sufficient_context_is_true_when_results_exist():
    """Any chunk that cleared the floor counts as sufficient."""
    chunk = TextChunk(chunk_id=0, text="Something", page_number=1)
    retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]

    assert retrieval_service.has_sufficient_context(retrieved) is True


def test_retrieve_rejects_a_blank_question():
    """A blank question would produce a meaningless embedding."""
    chunks = [TextChunk(chunk_id=0, text="Text", page_number=1)]

    model = FakeEmbedder({"Text": np.array([1.0, 0.0])})
    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    with pytest.raises(EmbeddingError, match="blank question"):
        retrieval_service.retrieve(
            question="   ",
            chunks=chunks,
            index=index,
            model=model,
            top_k=1,
            min_similarity=0.0,
        )


def test_retrieve_with_no_chunks_returns_empty():
    """An empty chunk list means no document text, so retrieval produces [].

    FAISS cannot build an index from zero vectors, so when a PDF yields no
    extractable text there is no index at all. ``retrieve`` must notice the empty
    chunk list and return before touching the index, so this test deliberately
    passes a one-vector index alongside ``chunks=[]`` to prove the chunk list is
    checked first.
    """
    dummy_model = FakeEmbedder({"x": np.array([1.0])})
    dummy_chunks = [TextChunk(chunk_id=0, text="x", page_number=1)]
    dummy_embeddings = embedding_service.embed_chunks(dummy_model, dummy_chunks)
    index = vector_store_service.build_index(dummy_embeddings)

    # Now call retrieve with chunks=[]
    results = retrieval_service.retrieve(
        question="anything",
        chunks=[],
        index=index,
        model=dummy_model,
        top_k=1,
        min_similarity=0.0,
    )

    assert results == []


def test_retrieve_scores_are_descending():
    """Results are ordered from highest score to lowest."""
    chunks = [
        TextChunk(chunk_id=0, text="A", page_number=1),
        TextChunk(chunk_id=1, text="B", page_number=2),
        TextChunk(chunk_id=2, text="C", page_number=3),
    ]

    # chunk 0: score 1.0
    # chunk 1: score 0.8
    # chunk 2: score 0.5
    query_vec = np.array([1.0, 0.0])
    chunk0_vec = np.array([1.0, 0.0])
    chunk1_vec = np.array([0.8, 0.6])  # normalised: (0.8, 0.6), dot with query = 0.8
    chunk2_vec = np.array([0.5, 0.866])  # normalised, dot with query = 0.5

    model = FakeEmbedder(
        {
            "q": query_vec,
            "A": chunk0_vec,
            "B": chunk1_vec,
            "C": chunk2_vec,
        }
    )

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    results = retrieval_service.retrieve(
        question="q",
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
        min_similarity=0.0,
    )

    # Verify descending order
    assert len(results) == 3
    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score

    # Verify specific values
    assert results[0].chunk.chunk_id == 0
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[1].chunk.chunk_id == 1
    assert results[1].score == pytest.approx(0.8, abs=1e-3)
    assert results[2].chunk.chunk_id == 2
    assert results[2].score == pytest.approx(0.5, abs=1e-3)


def test_retrieve_guards_the_chunk_id_invariant():
    """retrieve() asserts that chunks[row_index].chunk_id == row_index.

    Phase 2 numbered chunks sequentially from 0 in the same order they were
    embedded. If that invariant ever breaks, every citation would attach the wrong
    page number to the answer. This test proves the guard fires when the invariant
    is violated.
    """
    # Build chunks where chunk_id does NOT match the list position
    chunks = [
        TextChunk(chunk_id=5, text="A", page_number=1),  # chunk_id should be 0
        TextChunk(chunk_id=7, text="B", page_number=2),  # chunk_id should be 1
    ]

    query_vec = np.array([1.0, 0.0])
    chunk0_vec = np.array([1.0, 0.0])
    chunk1_vec = np.array([0.0, 1.0])

    model = FakeEmbedder(
        {
            "q": query_vec,
            "A": chunk0_vec,
            "B": chunk1_vec,
        }
    )

    embeddings = embedding_service.embed_chunks(model, chunks)
    index = vector_store_service.build_index(embeddings)

    # retrieve() should detect the mismatch and raise
    with pytest.raises(RetrievalError, match="chunk_id.*must match.*position"):
        retrieval_service.retrieve(
            question="q",
            chunks=chunks,
            index=index,
            model=model,
            top_k=2,
            min_similarity=0.0,
        )


def test_retrieval_error_is_a_runtime_error():
    """Phase 4 catches EmbeddingError/VectorStoreError/RetrievalError together.

    All three must share a base class, otherwise the UI's single ``except`` clause
    would let one of them through as a raw Streamlit traceback.
    """
    assert issubclass(RetrievalError, RuntimeError)


def test_retrieve_rejects_an_index_smaller_than_the_chunk_list():
    """A filtered-subset index with the full chunk list must be refused.

    This is the misalignment case the chunk_id guard cannot see. Only two of the
    three chunks were embedded, so index row 1 holds chunk 2's vector while
    chunks[1] is a different chunk. Because Phase 2 numbers chunks sequentially,
    chunks[1].chunk_id == 1 still holds and the chunk_id guard would pass, so
    every citation would be silently wrong. The length check must fire instead.
    """
    chunks = [
        TextChunk(chunk_id=0, text="A", page_number=1),
        TextChunk(chunk_id=1, text="B", page_number=2),
        TextChunk(chunk_id=2, text="C", page_number=3),
    ]

    model = FakeEmbedder(
        {
            "q": np.array([1.0, 0.0]),
            "A": np.array([1.0, 0.0]),
            "C": np.array([0.0, 1.0]),
        }
    )

    # Embed only chunks 0 and 2 - a two-vector index for a three-chunk list.
    embeddings = embedding_service.embed_chunks(model, [chunks[0], chunks[2]])
    index = vector_store_service.build_index(embeddings)
    assert index.ntotal == 2

    with pytest.raises(RetrievalError, match="2 vectors but 3 chunks"):
        retrieval_service.retrieve(
            question="q",
            chunks=chunks,
            index=index,
            model=model,
            top_k=2,
            min_similarity=0.0,
        )


def test_retrieve_rejects_an_index_larger_than_the_chunk_list():
    """A stale index left over from a longer document must be refused.

    Without the length check this path reached ``chunks[row_index]`` with a row
    index past the end of the list and raised a bare ``IndexError`` out of the
    service. It must raise ``RetrievalError`` with a clear message instead.
    """
    stale_chunks = [
        TextChunk(chunk_id=0, text="A", page_number=1),
        TextChunk(chunk_id=1, text="B", page_number=2),
        TextChunk(chunk_id=2, text="C", page_number=3),
    ]

    model = FakeEmbedder(
        {
            "q": np.array([1.0, 0.0]),
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([0.0, 1.0]),
        }
    )

    # The index describes three chunks, but only the first is passed to retrieve.
    embeddings = embedding_service.embed_chunks(model, stale_chunks)
    index = vector_store_service.build_index(embeddings)
    assert index.ntotal == 3

    with pytest.raises(RetrievalError, match="3 vectors but 1 chunks"):
        retrieval_service.retrieve(
            question="q",
            chunks=stale_chunks[:1],
            index=index,
            model=model,
            top_k=3,
            min_similarity=0.0,
        )
