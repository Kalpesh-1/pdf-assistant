"""Tests for the embedding and indexing pipeline as plain functions.

These tests validate that the pipeline builds a FAISS index that correctly aligns
with the chunk list, which is critical for citation accuracy. If the index row
order does not match the chunk list order, retrieved chunks would cite the wrong
page numbers.

No Streamlit or network calls here - a FakeEmbedder is injected to avoid
downloading the 90 MB model during test runs.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from pdf_assistant.models.schemas import PageText, TextChunk
from pdf_assistant.services.chunking_service import chunk_pages
from pdf_assistant.services.embedding_service import embed_chunks
from pdf_assistant.services.vector_store_service import build_index


class FakeEmbedder:
    """A test double that produces predictable embeddings without the real model.

    Each text is converted to a fixed-length vector from a checksum of its
    content. This makes the tests fast and avoids the 90 MB download and the
    several-second load time of the real SentenceTransformer.

    zlib.crc32 is used rather than the built-in hash(): Python salts string
    hashing differently in every process, so hash()-derived vectors would change
    from run to run and a failure could not be reproduced. crc32 is a fixed
    function of the bytes, so the vectors are identical on every run.
    """

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Produce a fake embedding for each text."""
        # Checksum the text, slice bits out of it to get a fixed number of
        # dimensions, then normalise. The exact values do not matter as long as
        # they are consistent and different texts produce different vectors.
        embeddings = []
        for text in texts:
            # Checksum to an integer, then convert to a fixed-length vector
            h = zlib.crc32(text.encode())
            # Create a 5-dimensional vector from the hash
            vec = np.array(
                [
                    (h % 1000) / 1000.0,
                    ((h >> 10) % 1000) / 1000.0,
                    ((h >> 20) % 1000) / 1000.0,
                    ((h >> 30) % 1000) / 1000.0,
                    ((h >> 40) % 1000) / 1000.0,
                ],
                dtype=np.float32,
            )
            # Normalise to unit length to match the real embedding service
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        return np.array(embeddings, dtype=np.float32)


def test_pipeline_builds_an_index_covering_every_chunk():
    """The index must hold exactly as many vectors as there are chunks.

    If the index held fewer, some chunks would be unretrievable. If it held more,
    the extra vectors would have no corresponding chunks, which would break
    citation lookup.
    """
    pages = [
        PageText(page_number=1, text="First page content here."),
        PageText(page_number=2, text="Second page with more text."),
        PageText(page_number=3, text="Third page concludes the document."),
    ]
    chunks = chunk_pages(pages, chunk_size=20, chunk_overlap=5)

    embedder = FakeEmbedder()
    vectors = embed_chunks(embedder, chunks)
    index = build_index(vectors)

    assert index.ntotal == len(chunks)


def test_index_row_order_matches_chunk_id_order():
    """The index must preserve chunk order for correct citation lookup.

    Retrieval returns a row index into the FAISS index. The application uses that
    row index to look up the corresponding chunk in the chunk list and extract its
    page_number for citation. If the index rows are in a different order from the
    chunks, the citation would point at the wrong page.

    This test embeds a list of chunks with known page numbers, searches for one of
    them, and confirms that the retrieved row index points back to the correct
    chunk in the original list.
    """
    chunks = [
        TextChunk(chunk_id=0, text="Introduction to the topic.", page_number=1),
        TextChunk(chunk_id=1, text="Detailed explanation follows.", page_number=2),
        TextChunk(chunk_id=2, text="Conclusion and summary.", page_number=3),
    ]

    embedder = FakeEmbedder()
    vectors = embed_chunks(embedder, chunks)
    index = build_index(vectors)

    # Embed the query and search for the top result
    query_text = "Detailed explanation follows."
    query_vector = embedder.encode([query_text], convert_to_numpy=True)

    # Import the search function to test the round-trip
    from pdf_assistant.services.vector_store_service import search

    results = search(index, query_vector, top_k=1)

    # The search should return row 1, which corresponds to chunks[1]
    assert len(results) == 1
    row_index, _score = results[0]
    assert row_index == 1
    assert chunks[row_index].text == query_text
    assert chunks[row_index].page_number == 2


def test_pipeline_on_a_blank_pdf_produces_no_index():
    """A PDF with no usable text should not build an index.

    The main pipeline already catches this and sets status to FAILED before
    reaching the embedding step, but this test documents that the embedding
    service itself raises EmbeddingError when given an empty chunk list.
    """
    from pdf_assistant.services.embedding_service import EmbeddingError

    chunks: list[TextChunk] = []

    embedder = FakeEmbedder()

    with pytest.raises(EmbeddingError) as exc_info:
        embed_chunks(embedder, chunks)

    assert "at least one" in str(exc_info.value).lower()
