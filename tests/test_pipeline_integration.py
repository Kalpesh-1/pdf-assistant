"""End-to-end tests for the extraction, chunking and retrieval pipeline.

The unit tests check extraction, chunking and retrieval separately. These tests
check the **joins** between them, which is where page numbers could be lost:

    PDF bytes -> validate -> extract per page -> clean -> chunk
    ... -> embed -> build index -> retrieve

If a page number were dropped or shifted anywhere along that path, every
citation the application shows would point at the wrong page while still looking
perfectly plausible. That is the worst kind of bug in this project, so it is
tested on the whole pipeline rather than on the pieces.

WHY THE RETRIEVAL HALF IS HERE TOO: every alignment test in
``tests/test_retrieval_service.py`` builds ``TextChunk(chunk_id=i, ...)`` by
hand, which *assumes* the very property under test. Only running the real
``chunk_pages`` output through ``embed_chunks`` and ``build_index`` proves that
the chunker, the vector array and the chunk list genuinely line up.

No network or API access: the embedding model is a deterministic fake keyed on
chunk text, so no 90 MB download happens.
"""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from pdf_assistant.services.chunking_service import chunk_pages
from pdf_assistant.services.embedding_service import embed_chunks, embed_query
from pdf_assistant.services.pdf_service import extract_pages, inspect_pdf, total_char_count
from pdf_assistant.services.retrieval_service import retrieve
from pdf_assistant.services.vector_store_service import build_index

TWENTY_MB = 20 * 1024 * 1024

# One unique marker word per page. Short enough that cleaning cannot alter them.
MARKERS = ("ALPHAMARK", "BETAMARK", "GAMMAMARK", "DELTAMARK")


@pytest.fixture
def four_page_wordy_pdf() -> bytes:
    """A four-page PDF with enough text per page to force several chunks.

    Each page repeats its own marker word, so the page a chunk came from can be
    identified from the chunk text alone.
    """
    document = pymupdf.open()
    for marker in MARKERS:
        page = document.new_page()
        # Many short lines rather than one long line, because PyMuPDF does not
        # wrap text and anything past the page edge would not be extracted.
        sentences = [f"{marker} appears in sentence number {n} of this page." for n in range(40)]
        page.insert_text((40, 60), "\n".join(sentences), fontsize=8)
    data = document.tobytes()
    document.close()
    return data


def _run_pipeline(file_bytes: bytes, chunk_size: int = 400, chunk_overlap: int = 80):
    """Run validate -> extract -> chunk, the way ``app.py`` does."""
    result, metadata = inspect_pdf(file_bytes, "sample.pdf", TWENTY_MB)
    assert result.is_valid, result.message
    assert metadata is not None

    pages = extract_pages(file_bytes)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return metadata, pages, chunks


def test_pipeline_produces_pages_and_chunks(four_page_wordy_pdf):
    metadata, pages, chunks = _run_pipeline(four_page_wordy_pdf)

    assert metadata.total_pages == 4
    assert len(pages) == 4
    # The fixture is wordy enough that each page must split into several chunks.
    assert len(chunks) > 4


def test_every_chunk_is_attributed_to_the_page_it_came_from(four_page_wordy_pdf):
    """The core guarantee: a chunk's page number matches its actual source page.

    Marker word N only ever appears on page N, so any chunk containing it must
    carry that page number.
    """
    _, _, chunks = _run_pipeline(four_page_wordy_pdf)

    for chunk in chunks:
        found = [marker for marker in MARKERS if marker in chunk.text]
        assert found, f"chunk {chunk.chunk_id} contains no marker, cannot be verified"
        # A chunk must never mix text from two pages.
        assert len(set(found)) == 1, (
            f"chunk {chunk.chunk_id} on page {chunk.page_number} mixes pages: {found}"
        )
        expected_page = MARKERS.index(found[0]) + 1
        assert chunk.page_number == expected_page


def test_page_numbers_are_one_based_and_cover_every_page(four_page_wordy_pdf):
    _, pages, chunks = _run_pipeline(four_page_wordy_pdf)

    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert sorted({chunk.page_number for chunk in chunks}) == [1, 2, 3, 4]


def test_chunk_ids_are_unique_and_sequential_across_the_document(four_page_wordy_pdf):
    _, _, chunks = _run_pipeline(four_page_wordy_pdf)

    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == list(range(len(chunks))), "chunk_id must index into the vector array"


def test_character_count_matches_the_extracted_pages(four_page_wordy_pdf):
    _, pages, _ = _run_pipeline(four_page_wordy_pdf)

    assert total_char_count(pages) == sum(len(page.text) for page in pages)
    assert total_char_count(pages) > 0


def test_no_chunk_exceeds_the_configured_size(four_page_wordy_pdf):
    _, _, chunks = _run_pipeline(four_page_wordy_pdf, chunk_size=300, chunk_overlap=50)

    assert all(chunk.char_count <= 300 for chunk in chunks)


def test_chunk_text_really_exists_in_its_cited_page(four_page_wordy_pdf):
    """A citation must point at text that is genuinely on that page.

    Chunking can insert nothing and reorder nothing, so a sample of words from
    each chunk must be findable in the page it claims to come from.
    """
    _, pages, chunks = _run_pipeline(four_page_wordy_pdf)
    # Whitespace is normalised on both sides. The page text keeps its line
    # breaks, while a chunk may start or end mid-line, so a byte-exact substring
    # check would fail on formatting rather than on wrong content.
    page_text_by_number = {page.page_number: " ".join(page.text.split()) for page in pages}

    for chunk in chunks:
        source = page_text_by_number[chunk.page_number]
        sample = " ".join(chunk.text.split()[1:6])
        assert sample in source, (
            f"chunk {chunk.chunk_id} text not found on page {chunk.page_number}"
        )


def test_blank_pdf_extracts_pages_but_produces_no_chunks(blank_pdf):
    """A scanned-style PDF yields numbered pages with no text, so no chunks.

    ``app.py`` relies on this to show an OCR-not-supported message instead of
    building an empty index.
    """
    pages = extract_pages(blank_pdf)
    chunks = chunk_pages(pages, chunk_size=400, chunk_overlap=80)

    assert len(pages) == 2
    assert chunks == []
    assert total_char_count(pages) == 0


def test_marker_pdf_from_conftest_flows_through_correctly(three_page_pdf):
    """The shared three-page fixture: APPLE/1, BANANA/2, CHERRY/3."""
    _, _, chunks = _run_pipeline(three_page_pdf)

    pages_by_marker = {
        marker: {chunk.page_number for chunk in chunks if marker in chunk.text}
        for marker in ("APPLE", "BANANA", "CHERRY")
    }

    assert pages_by_marker == {"APPLE": {1}, "BANANA": {2}, "CHERRY": {3}}


# --------------------------------------------------------------------------
# The retrieval half of the pipeline, over REAL chunker output.
# --------------------------------------------------------------------------

# The three marker words from the shared three_page_pdf fixture, in page order.
MARKER_VOCABULARY = ("APPLE", "BANANA", "CHERRY")


class MarkerEmbedder:
    """Deterministic stand-in for the sentence-transformer model.

    Dimension 0 is a small constant baseline, so no text ever produces an
    all-zero vector. Dimensions 1..3 record whether each marker word appears in
    the text. Two texts sharing a marker word therefore score close to 1.0
    against each other, while texts with no marker in common score close to 0.

    That is enough to prove the retrieval wiring without downloading the real
    90 MB model and without any randomness: a question about BANANA must come
    back with the chunk that genuinely contains BANANA.
    """

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        """Return one vector per text, derived only from the text's content."""
        rows = []
        for text in texts:
            upper = text.upper()
            row = [0.1] + [1.0 if marker in upper else 0.0 for marker in MARKER_VOCABULARY]
            rows.append(row)
        return np.array(rows, dtype=np.float32)


def test_real_chunker_output_matches_the_index_row_for_row(three_page_pdf):
    """The index must hold exactly one vector per chunk the chunker produced.

    Every alignment test in test_retrieval_service.py hand-builds
    TextChunk(chunk_id=i, ...), which assumes the property it is checking. This
    test runs the genuine chunk_pages -> embed_chunks -> build_index path, so a
    chunker change that dropped or duplicated a chunk would be caught here.
    """
    _, _, chunks = _run_pipeline(three_page_pdf)
    model = MarkerEmbedder()

    vectors = embed_chunks(model, chunks)
    index = build_index(vectors)

    assert chunks, "the fixture must produce chunks for this test to mean anything"
    assert vectors.shape[0] == len(chunks)
    assert index.ntotal == len(chunks)
    # The same sequential numbering retrieve() relies on, from the real chunker.
    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_retrieval_over_real_chunks_cites_the_page_the_text_is_really_on(three_page_pdf):
    """The full round trip: a BANANA question must cite page 2, provably.

    This asserts both halves of a citation. The page number must be 2, AND the
    retrieved text must genuinely appear in the text extracted from page 2 - so
    a misaligned index that returned a plausible-looking page number would still
    fail here.
    """
    _, pages, chunks = _run_pipeline(three_page_pdf)
    model = MarkerEmbedder()

    vectors = embed_chunks(model, chunks)
    index = build_index(vectors)

    results = retrieve(
        question="Tell me about BANANA",
        chunks=chunks,
        index=index,
        model=model,
        top_k=1,
        min_similarity=0.5,
    )

    assert len(results) == 1
    best = results[0].chunk

    # Half one: the citation points at page 2.
    assert best.page_number == 2

    # Half two: the cited text is really on page 2. Whitespace is normalised on
    # both sides because the page text keeps its line breaks while a chunk may
    # start or end mid-line.
    page_two_text = next(" ".join(page.text.split()) for page in pages if page.page_number == 2)
    assert " ".join(best.text.split()) in page_two_text
    assert "BANANA" in best.text


def test_retrieval_over_real_chunks_abstains_on_an_unrelated_question(three_page_pdf):
    """A question about nothing in the document returns [], the abstention signal."""
    _, _, chunks = _run_pipeline(three_page_pdf)
    model = MarkerEmbedder()

    index = build_index(embed_chunks(model, chunks))

    results = retrieve(
        question="What is the share price of a quantum submarine?",
        chunks=chunks,
        index=index,
        model=model,
        top_k=3,
        min_similarity=0.5,
    )

    assert results == []


def test_query_embedding_width_matches_the_chunk_embedding_width(three_page_pdf):
    """Query and chunk vectors must be the same width or FAISS cannot search.

    Checked over real chunker output rather than hand-built chunks, because a
    width mismatch here is what would make the whole retrieval step unusable.
    """
    _, _, chunks = _run_pipeline(three_page_pdf)
    model = MarkerEmbedder()

    chunk_vectors = embed_chunks(model, chunks)
    query_vector = embed_query(model, "Tell me about BANANA")

    assert query_vector.shape == (1, chunk_vectors.shape[1])
