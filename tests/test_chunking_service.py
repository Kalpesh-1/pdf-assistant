"""Tests for the text chunker.

The important property being tested here is not "does it split text" but
"does every chunk still know which page it came from". That is what the
citations shown under an answer are built from, so a chunk with the wrong page
number would make the application lie to the user.
"""

from __future__ import annotations

import pytest

from pdf_assistant.models.schemas import PageText
from pdf_assistant.services.chunking_service import chunk_pages

# A distinct word per page. If a chunk contains BANANA it must have come from
# page 2, so its page_number must be 2 - that is the whole point of the marker.
MARKERS = {1: "APPLE", 2: "BANANA", 3: "CHERRY"}


def _page_text(marker: str, sentences: int) -> str:
    """Build believable prose containing ``marker`` in every sentence."""
    return " ".join(
        f"Sentence number {index} of this page mentions {marker} quite clearly."
        for index in range(1, sentences + 1)
    )


def _marked_pages(sentences: int = 12) -> list[PageText]:
    """Three pages of prose, each tagged with its own marker word."""
    return [
        PageText(page_number=number, text=_page_text(marker, sentences))
        for number, marker in MARKERS.items()
    ]


# ---------------------------------------------------------------------------
# Basic splitting behaviour
# ---------------------------------------------------------------------------


def test_short_page_produces_one_chunk_with_the_whole_text():
    text = "This page is comfortably shorter than the chunk size."
    pages = [PageText(page_number=1, text=text)]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == 0


def test_long_page_produces_several_chunks_within_the_size_limit():
    pages = [PageText(page_number=1, text=_page_text("APPLE", sentences=20))]

    chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 150 for chunk in chunks)
    # Nothing should be lost: the first chunk starts at the start of the page
    # and the last chunk ends at the end of it.
    assert pages[0].text.startswith(chunks[0].text[:40])
    assert pages[0].text.endswith(chunks[-1].text[-40:])


def test_every_chunk_covers_the_whole_page_text_in_order():
    """Walking the chunks in order must march forward through the page."""
    page_text = _page_text("APPLE", sentences=15)
    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 120, 30)

    position = 0
    for chunk in chunks:
        found = page_text.find(chunk.text, position)
        assert found != -1, f"chunk text not found in the page after position {position}"
        # Chunks may overlap, but a later chunk never starts before an earlier one.
        assert found >= position
        position = found


# ---------------------------------------------------------------------------
# Page metadata preservation - the reason this module exists
# ---------------------------------------------------------------------------


def test_page_number_matches_the_page_the_text_came_from():
    pages = _marked_pages()

    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=30)

    assert chunks, "three pages of prose should produce chunks"
    for chunk in chunks:
        expected_marker = MARKERS[chunk.page_number]
        assert expected_marker in chunk.text, (
            f"chunk {chunk.chunk_id} claims page {chunk.page_number} but does not "
            f"contain that page's marker {expected_marker}"
        )


def test_no_chunk_mixes_text_from_two_pages():
    """A chunk must never span a page boundary, or its citation is a guess."""
    pages = _marked_pages()

    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=30)

    for chunk in chunks:
        foreign = [
            marker
            for page_number, marker in MARKERS.items()
            if page_number != chunk.page_number and marker in chunk.text
        ]
        assert not foreign, f"chunk {chunk.chunk_id} also contains text from {foreign}"


def test_all_three_pages_are_represented():
    chunks = chunk_pages(_marked_pages(), chunk_size=120, chunk_overlap=30)

    assert sorted({chunk.page_number for chunk in chunks}) == [1, 2, 3]
    # Several chunks per page, so the test above is not passing by accident on
    # a single chunk per page.
    assert len(chunks) > 3


def test_page_numbers_survive_an_empty_page_in_the_middle():
    """Skipping page 2 must not shift page 3's chunks onto page 2."""
    pages = [
        PageText(page_number=1, text=_page_text("APPLE", sentences=4)),
        PageText(page_number=2, text=""),
        PageText(page_number=3, text=_page_text("CHERRY", sentences=4)),
    ]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=25)

    assert sorted({chunk.page_number for chunk in chunks}) == [1, 3]
    for chunk in chunks:
        assert MARKERS[chunk.page_number] in chunk.text


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def test_consecutive_chunks_of_the_same_page_overlap():
    """The tail of one chunk must reappear at the start of the next one.

    This is what stops a sentence sitting on a chunk boundary from being torn
    in half and becoming impossible to retrieve.
    """
    overlap = 60
    pages = [PageText(page_number=1, text=_page_text("APPLE", sentences=20))]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=overlap)

    assert len(chunks) >= 3, "need several chunks for this test to mean anything"
    for index in range(len(chunks) - 1):
        # 20 characters is comfortably inside the 60-character overlap, so it
        # must be repeated even after the chunks are stripped of whitespace.
        tail = chunks[index].text[-20:]
        following = chunks[index + 1]
        assert tail in following.text, f"chunk {following.chunk_id} does not repeat {tail!r}"


def test_zero_overlap_is_allowed_and_repeats_nothing():
    page_text = _page_text("APPLE", sentences=20)

    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 150, 0)

    assert len(chunks) > 1
    # With no overlap the chunks are disjoint, so their lengths cannot add up
    # to more than the page itself.
    assert sum(chunk.char_count for chunk in chunks) <= len(page_text)


# ---------------------------------------------------------------------------
# chunk_id numbering
# ---------------------------------------------------------------------------


def test_chunk_ids_are_sequential_from_zero_across_the_whole_document():
    """ids are the positions in the FAISS array, so they must not restart."""
    chunks = chunk_pages(_marked_pages(), chunk_size=120, chunk_overlap=30)

    ids = [chunk.chunk_id for chunk in chunks]

    assert ids == list(range(len(chunks)))
    assert len(set(ids)) == len(ids)
    assert ids[0] == 0


# ---------------------------------------------------------------------------
# Empty and whitespace-only input
# ---------------------------------------------------------------------------


def test_no_pages_returns_no_chunks():
    assert chunk_pages([], chunk_size=100, chunk_overlap=20) == []


def test_empty_and_whitespace_only_pages_produce_no_chunks():
    pages = [
        PageText(page_number=1, text=""),
        PageText(page_number=2, text="   \n\n \t  \n "),
        PageText(page_number=3, text="\n"),
    ]

    assert chunk_pages(pages, chunk_size=100, chunk_overlap=20) == []


def test_no_chunk_is_empty_or_whitespace_only():
    pages = [
        PageText(page_number=1, text="Real text on page one.\n\n\n"),
        PageText(page_number=2, text="    "),
        PageText(page_number=3, text=_page_text("CHERRY", sentences=6)),
    ]

    chunks = chunk_pages(pages, chunk_size=80, chunk_overlap=20)

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.text == chunk.text.strip() for chunk in chunks)


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_chunk_size_below_one_raises(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        chunk_pages([PageText(page_number=1, text="Some text.")], chunk_size, 0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_pages([PageText(page_number=1, text="Some text.")], 100, -1)


@pytest.mark.parametrize(("chunk_size", "chunk_overlap"), [(100, 100), (100, 150)])
def test_overlap_not_smaller_than_chunk_size_raises(chunk_size, chunk_overlap):
    """Equal sizes would leave the window standing still, so it is rejected."""
    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        chunk_pages([PageText(page_number=1, text="Some text.")], chunk_size, chunk_overlap)


def test_arguments_are_validated_before_any_work_is_done():
    """The error is raised even when there is nothing at all to chunk."""
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=10, chunk_overlap=10)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def test_terminates_with_an_almost_total_overlap():
    """chunk_size=10 with chunk_overlap=9 is the worst legal case.

    The window can only step forward by one character, so the chunk count is
    bounded by the length of the text. The assertion on the count is the real
    check: a chunker that failed to guarantee forward progress would either
    hang or produce far more chunks than there are characters.
    """
    page_text = _page_text("APPLE", sentences=10)
    pages = [PageText(page_number=1, text=page_text)]

    chunks = chunk_pages(pages, chunk_size=10, chunk_overlap=9)

    assert 1 <= len(chunks) <= len(page_text)
    assert all(chunk.char_count <= 10 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_terminates_on_text_with_no_boundaries_at_all():
    """A long run with no spaces leaves nowhere tidy to cut, so it hard-cuts."""
    page_text = "x" * 500
    pages = [PageText(page_number=1, text=page_text)]

    chunks = chunk_pages(pages, chunk_size=10, chunk_overlap=9)

    assert 1 <= len(chunks) <= len(page_text)
    assert all(chunk.char_count <= 10 for chunk in chunks)


def test_single_character_chunk_size_still_terminates():
    chunks = chunk_pages([PageText(page_number=1, text="abcdef")], 1, 0)

    assert [chunk.text for chunk in chunks] == list("abcdef")


# ---------------------------------------------------------------------------
# Cut quality
# ---------------------------------------------------------------------------


def test_words_are_not_sliced_in_half_in_normal_prose():
    page_text = _page_text("APPLE", sentences=25)
    vocabulary = set(page_text.split())

    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 200, 50)

    assert len(chunks) > 1
    for chunk in chunks:
        broken = [word for word in chunk.text.split() if word not in vocabulary]
        assert not broken, f"chunk {chunk.chunk_id} contains sliced words: {broken}"


def test_no_word_of_the_page_is_lost():
    """Nudging the next chunk to a word start must not drop any text."""
    page_text = _page_text("APPLE", sentences=25)

    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 200, 50)

    chunked_words = {word for chunk in chunks for word in chunk.text.split()}
    assert set(page_text.split()) == chunked_words


def test_a_paragraph_break_is_preferred_as_the_cut_point():
    first = "A" * 90 + "."
    second = "B" * 90 + "."
    page_text = f"{first}\n\n{second}"

    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 120, 20)

    # The paragraph break sits 91 characters in, inside the searched last third
    # of a 120-character window, so the first chunk should stop exactly there.
    assert chunks[0].text == first
    assert chunks[-1].text.endswith(second[-20:])


def test_a_sentence_end_is_preferred_when_there_is_no_paragraph_break():
    page_text = "First sentence is here. " + "word " * 40

    chunks = chunk_pages([PageText(page_number=1, text=page_text)], 30, 5)

    # The full stop is 23 characters in, past two thirds of a 30-character
    # window, so the first chunk should be exactly that sentence.
    assert chunks[0].text == "First sentence is here."
