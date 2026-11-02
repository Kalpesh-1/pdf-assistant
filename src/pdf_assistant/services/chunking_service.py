"""Splitting cleaned page text into overlapping chunks for retrieval.

This is the last step of PDF processing and the step that decides how good
retrieval can possibly be. The embedding model turns each chunk into one
vector, so a chunk is the smallest unit the search can return. Chunks that are
too large dilute the meaning of that single vector; chunks that are too small
lose the context the language model needs to answer.

Two decisions in this module are worth understanding before changing anything,
because the correctness of every citation in the application depends on them.
They are explained in :func:`chunk_pages`.
"""

from __future__ import annotations

from pdf_assistant.models.schemas import PageText, TextChunk

# Boundaries the chunker prefers to cut at, in order of preference.
#
# A paragraph break is the best cut: the text either side of it is already
# understood to be two separate thoughts.
_PARAGRAPH_BREAK = "\n\n"

# A sentence end is the next best cut. The trailing space or newline is part of
# the pattern on purpose - it is what distinguishes the full stop ending
# "...in 1947. The next" from the full stop inside "Fig. 2" or "3.5 marks".
_SENTENCE_ENDINGS = (". ", "! ", "? ", ".\n", "!\n", "?\n")

# Fraction of a chunk that must be used up before a boundary is accepted.
#
# Without this, a paragraph break sitting 10 characters into an 800-character
# window would be taken, producing a 10-character chunk and many more chunks
# than necessary. Searching only the last third of the window keeps chunks
# reasonably full while still allowing a tidy cut.
_MIN_FILL_FRACTION = 2 / 3


def chunk_pages(
    pages: list[PageText],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split the cleaned text of every page into overlapping chunks.

    **Chunks never cross a page boundary.** Each page is chunked on its own,
    and a chunk is always made entirely of text from a single page. This is a
    deliberate design decision, not a limitation:

    * Every chunk therefore has exactly one correct ``page_number``. A chunk
      that spanned the end of page 4 and the start of page 5 would have no
      honest answer to "which page is this from?", and the citation shown to
      the user would be a guess.
    * The cost is small. A sentence split across a page break may end up in two
      different chunks, but each of those chunks is still retrievable and still
      correctly attributed.

    The alternative - joining the whole document into one long string and
    chunking that - produces slightly better-formed chunks and untrustworthy
    citations. For this project, trustworthy citations matter more.

    **Consecutive chunks of the same page overlap** by ``chunk_overlap``
    characters. Overlap exists because a chunk boundary is an arbitrary cut: a
    sentence that happens to sit across it would be torn in half, and neither
    half on its own would look similar enough to the user's question for the
    vector search to find it. Repeating the tail of one chunk at the start of
    the next means every sentence appears whole in at least one chunk.

    Args:
        pages: Cleaned page text, in document order, as produced by the PDF
            service. Pages with no text are skipped.
        chunk_size: Maximum number of characters in one chunk.
        chunk_overlap: Number of characters repeated between consecutive chunks
            of the same page. Must be smaller than ``chunk_size``.

    Returns:
        The chunks of the whole document, in page order. ``chunk_id`` is
        numbered from 0 across the entire document (not restarted per page),
        because in Phase 3 it is used as the position of the chunk's vector in
        the FAISS index - that index is one flat array for the whole document.

    Raises:
        ValueError: If ``chunk_size`` is below 1, ``chunk_overlap`` is negative,
            or ``chunk_overlap`` is greater than or equal to ``chunk_size``. The
            last case is rejected because the next window starts
            ``chunk_size - chunk_overlap`` characters further on, so an overlap
            as large as the chunk would make that step zero or negative and the
            chunker would never reach the end of the page.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, but got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, but got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
            f"({chunk_size}), otherwise chunking cannot make progress."
        )

    chunks: list[TextChunk] = []

    for page in pages:
        # A page can legitimately be empty - a cover page, a page that held only
        # an image, or a deliberately blank page. There is nothing to retrieve
        # from it, so it contributes no chunks. Its page number is simply never
        # used, which keeps the remaining page numbers correct.
        page_text = page.text.strip()
        if not page_text:
            continue

        for piece in _split_page_text(page_text, chunk_size, chunk_overlap):
            chunks.append(
                TextChunk(
                    # len(chunks) is the position this chunk will occupy, which
                    # is exactly the sequential, document-wide id we want.
                    chunk_id=len(chunks),
                    text=piece,
                    page_number=page.page_number,
                )
            )

    return chunks


def _split_page_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one page's text into overlapping pieces of at most ``chunk_size``.

    The caller has already validated the two size arguments and stripped the
    text, so this function only has to do the sliding-window walk.
    """
    pieces: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        limit = start + chunk_size

        if limit >= length:
            # The rest of the page fits in one chunk, so no cut is needed.
            end = length
        else:
            # Relative position returned by the boundary search, converted back
            # to an absolute position in the page text.
            end = start + _find_cut(text[start:limit])

        piece = text[start:end].strip()
        if piece:
            # Whitespace-only windows (a run of blank lines) add nothing to the
            # index, so they are dropped rather than stored as empty chunks.
            pieces.append(piece)

        if end >= length:
            break

        # Step back by the overlap so the next chunk repeats the tail of this
        # one. max(..., start + 1) is the termination guarantee: with a large
        # overlap and a small chunk the plain subtraction could land on or
        # before the current start, and the loop would never finish. Advancing
        # by at least one character means the walk always reaches the end.
        next_start = max(end - chunk_overlap, start + 1)

        # The subtraction above is pure arithmetic, so it usually lands in the
        # middle of a word. Nudge it forward to the start of a whole word.
        start = _next_word_start(text, next_start, limit=end)

    return pieces


def _next_word_start(text: str, position: int, limit: int) -> int:
    """Move ``position`` forward to the start of a whole word.

    Stepping back by the overlap lands on an arbitrary character, which is
    usually inside a word, so the next chunk would begin with a fragment such
    as ``"mber"`` instead of ``"number"``. A fragment is noise: it means
    nothing to the embedding model and looks wrong in a source excerpt.

    Args:
        text: The page text being walked.
        position: Where the next chunk would otherwise start.
        limit: The end of the previous chunk. The search never goes past this,
            which is what makes the adjustment safe - every character skipped
            is already contained in the previous chunk, so no text is lost.

    Returns:
        A position between ``position`` and ``limit``, so the walk still only
        ever moves forward.
    """
    if position <= 0 or position >= limit:
        return position
    if text[position - 1].isspace():
        return position  # Already at a word start.

    for index in range(position, limit):
        if text[index].isspace():
            return index + 1

    # The whole overlap is one unbroken run of characters (a long URL, say), so
    # there is no word start to move to. Leave the position as it was.
    return position


def _find_cut(window: str) -> int:
    """Choose where to end a chunk inside ``window``.

    Searches backwards from the end of the window for a natural boundary so a
    chunk does not stop in the middle of a word: first a paragraph break, then
    a sentence end, then any whitespace. Only the last part of the window is
    searched (see ``_MIN_FILL_FRACTION``).

    Args:
        window: The candidate chunk, already cut to the maximum length.

    Returns:
        The number of characters to take from ``window``. Always at least 1, so
        the caller is guaranteed to move forward.
    """
    # Earliest offset a boundary is allowed at. max(..., 1) keeps the returned
    # length above zero even for a one-character window.
    earliest = max(int(len(window) * _MIN_FILL_FRACTION), 1)

    paragraph = window.rfind(_PARAGRAPH_BREAK, earliest)
    if paragraph != -1:
        # Cut after the break so the blank line is not carried into the chunk.
        return paragraph + len(_PARAGRAPH_BREAK)

    # The latest sentence end wins, because it uses the most of the window.
    sentence_cut = -1
    for ending in _SENTENCE_ENDINGS:
        found = window.rfind(ending, earliest)
        if found != -1:
            sentence_cut = max(sentence_cut, found + len(ending))
    if sentence_cut != -1:
        return sentence_cut

    whitespace = max(window.rfind(" ", earliest), window.rfind("\n", earliest))
    if whitespace != -1:
        return whitespace + 1

    # No boundary at all in the searched range, so cut at the limit. This is
    # rare in prose but does happen with a long URL, a chemical name or a table
    # row with no spaces. A hard cut is better than refusing to split, because
    # a chunk longer than chunk_size would break the size guarantee.
    return len(window)
