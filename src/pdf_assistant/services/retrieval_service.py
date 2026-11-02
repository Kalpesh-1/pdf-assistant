"""Join embedding and search into retrieval with a similarity floor.

This is the fourth step of the RAG pipeline and the last before the language model
sees anything. Phase 2 extracted and chunked the PDF; Task 1 turned chunks into
vectors; Task 2 built the searchable index. Now the user's question is embedded,
the index returns the most similar chunks, and the similarity floor decides whether
those chunks are good enough to base an answer on.

WHY THE SIMILARITY FLOOR MATTERS: without it, the retrieval step always returns
*something*, even when the document has nothing relevant. The language model would
then see irrelevant chunks and either hallucinate an answer or hedge in unhelpful
ways. With the floor, weak matches are filtered out at retrieval time, and the
application can detect the empty result and abstain with a clear message instead.

The floor is configurable because what counts as "relevant enough" depends on the
document and the question. A technical manual might need high scores (0.4+) because
off-topic matches are dangerous. A narrative document might work with lower scores
(0.2+) because questions are often paraphrases. Phase 5 evaluation will tune this.
"""

from __future__ import annotations

from typing import Any

from pdf_assistant.models.schemas import RetrievedChunk, TextChunk
from pdf_assistant.services import embedding_service, vector_store_service

# EXCEPTIONS RAISED BY THIS MODULE (one place for Phase 4's UI to look):
#   - RetrievalError  - this module's own invariant failures (index/chunk mismatch).
#   - EmbeddingError  - re-raised from embedding_service (blank question, model failure).
#   - VectorStoreError - re-raised from vector_store_service (bad search input).
# All three subclass RuntimeError, so the UI can catch
# (RetrievalError, EmbeddingError, VectorStoreError) - or just RuntimeError - and
# be sure nothing escapes as a raw Streamlit traceback.


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot be trusted to produce correct citations.

    This is its own class, rather than a bare ``RuntimeError``, so the Phase 4 UI
    can catch it alongside ``EmbeddingError`` and ``VectorStoreError`` in a single
    ``except`` clause. A bare ``RuntimeError`` would slip past such a clause and
    surface to the student as an unhandled Streamlit traceback.
    """


def retrieve(
    question: str,
    chunks: list[TextChunk],
    index: Any,
    model: Any,
    top_k: int,
    min_similarity: float,
) -> list[RetrievedChunk]:
    """Retrieve the chunks most similar to the question, filtered by score.

    Args:
        question: The user's question. Must not be blank.
        chunks: All chunks from the document, in the same order they were
            embedded. This list is used to map search results (which are row
            indices into the embedding array) back to the original chunks and
            their page numbers.
        index: A FAISS index built by ``vector_store_service.build_index``.
        model: An object with an ``.encode`` method. In production this is a
            SentenceTransformer; in tests it is a fake.
        top_k: How many nearest neighbours to retrieve from the index before
            applying the similarity floor.
        min_similarity: Score threshold in [0.0, 1.0]. Only chunks scoring
            >= this value are returned. When no chunks clear the floor, the
            result is an empty list, which signals that the application should
            abstain rather than guess.

    Returns:
        A list of :class:`RetrievedChunk` objects, ordered from highest score to
        lowest, containing only chunks that scored >= ``min_similarity``. Length
        is between 0 (nothing relevant) and ``min(top_k, len(chunks))`` (all
        candidates cleared the floor).

    Raises:
        EmbeddingError: If the question is blank or embedding fails.
        VectorStoreError: If search fails.
        RetrievalError: If the index and the chunk list do not describe the same
            document - either because their lengths differ or because the
            ``chunks[row_index].chunk_id == row_index`` invariant is violated.
            Both guards prevent silent citation errors.
    """
    # Edge case: no chunks means no index was built (document is empty or
    # processing failed). Return [] immediately to avoid searching an invalid
    # index or accessing chunks[row_index] on an empty list.
    if not chunks:
        return []

    # CRITICAL INVARIANT 1 (length): the index must hold exactly one vector per
    # chunk. The chunk_id check further down catches *renumbering*, but not
    # *misalignment*: if a caller embedded a filtered subset of chunks and then
    # passed the full list here, row 3 of the index would be one chunk while
    # chunks[3] is a different one. Because Phase 2 numbers chunks sequentially,
    # chunks[3].chunk_id == 3 would still hold, the chunk_id guard would pass,
    # and every citation would be silently wrong. A stale index with MORE
    # vectors than chunks would instead raise a bare IndexError from inside this
    # service. Comparing the lengths catches both cases up front.
    if index.ntotal != len(chunks):
        raise RetrievalError(
            f"The vector index holds {index.ntotal} vectors but {len(chunks)} chunks were "
            f"supplied. The index and the chunk list must describe the same document; "
            f"otherwise citations would point at the wrong pages."
        )

    # Embed the question. This raises EmbeddingError on a blank question, which
    # is the intended behaviour - retrieval is meaningless without a question.
    query_vector = embedding_service.embed_query(model, question)

    # Search the index for the top_k nearest neighbours.
    raw_results = vector_store_service.search(index, query_vector, top_k)

    # Map (row_index, score) to RetrievedChunk, asserting the critical invariant
    # and filtering by the similarity floor.
    retrieved = []
    for row_index, score in raw_results:
        # CRITICAL INVARIANT 2 (numbering): row_index must equal chunk_id because
        # Phase 2 numbered chunks sequentially from 0 in the same order they were
        # embedded. If this ever drifts, every citation in the application would
        # attach the wrong page number to the answer. That is the worst failure
        # this project can produce, so we assert it rather than trusting it.
        chunk = chunks[row_index]
        if chunk.chunk_id != row_index:
            raise RetrievalError(
                f"Chunk at position {row_index} has chunk_id {chunk.chunk_id}, "
                f"but chunk_id must match its position in the list. This invariant "
                f"violation would produce incorrect page citations."
            )

        # Apply the similarity floor. Weak matches are dropped here rather than
        # being passed to the language model, which would either hallucinate or
        # hedge unhelpfully.
        if score >= min_similarity:
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

    # raw_results is already ordered by score descending (vector_store_service
    # guarantees this), and we preserved that order, so no explicit sort needed.
    return retrieved


def has_sufficient_context(retrieved: list[RetrievedChunk]) -> bool:
    """Check whether enough relevant chunks were found to answer the question.

    Args:
        retrieved: The result of :func:`retrieve`, possibly empty.

    Returns:
        ``True`` when at least one chunk cleared the similarity floor, ``False``
        otherwise. The caller uses this to decide between calling the language
        model (sufficient context) and abstaining with a clear message (no
        relevant chunks found).

    Note:
        This function exists rather than the caller checking ``len(retrieved) > 0``
        directly because the definition of "sufficient" might evolve. For example,
        a future version might require a minimum combined score across all chunks,
        or at least two chunks for controversial topics. Centralising the logic
        here makes that change safe.
    """
    return len(retrieved) > 0
