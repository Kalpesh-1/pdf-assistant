"""Data structures passed between the services and the user interface.

These are plain dataclasses, not database models. They exist so that each
service has a clear, typed input and output instead of passing loose
dictionaries around.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum


class ProcessingStatus(str, Enum):
    """Where the uploaded document currently is in the pipeline.

    The Overview and Document Details sections display this so the user can see
    that work is happening instead of staring at a blank screen.
    """

    NO_DOCUMENT = "No document uploaded"
    VALIDATING = "Validating PDF"
    EXTRACTING = "Extracting text"
    CHUNKING = "Splitting text into chunks"
    # Text is extracted and chunked, but the vector index does not exist yet, so
    # questions cannot be answered. Reached at the end of Phase 2, and a brief
    # transient state in the full pipeline. It is also where a document lands if
    # the embedding step later fails.
    TEXT_READY = "Text extracted and chunked"
    EMBEDDING = "Creating embeddings"
    INDEXING = "Building vector index"
    READY = "Ready"
    FAILED = "Failed"


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page.

    ``page_number`` is the page number as the reader sees it in the PDF, i.e.
    1-based. PyMuPDF indexes pages from 0, so the conversion happens once during
    extraction and the real page number is carried everywhere after that. This
    is what makes honest citations possible.
    """

    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class TextChunk:
    """One retrievable piece of the document, with its source page.

    The ``page_number`` here is copied from the :class:`PageText` the chunk came
    from. When a chunk is retrieved to answer a question, this field is what the
    citation is built from, in application code. The language model is never
    asked to supply a page number.
    """

    chunk_id: int
    text: str
    page_number: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector search, together with its score."""

    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class DocumentMetadata:
    """Facts about the uploaded file, shown in Overview and Document Details."""

    filename: str
    size_bytes: int
    total_pages: int

    # Filled in after extraction and chunking. Zero until those steps run.
    char_count: int = 0
    chunk_count: int = 0

    def with_counts(self, char_count: int, chunk_count: int) -> DocumentMetadata:
        """Return a copy carrying the extraction and chunking totals.

        This dataclass is frozen, so the pipeline builds a new object rather
        than mutating the existing one. That keeps the metadata in session state
        from being changed by accident halfway through processing.
        """
        return replace(self, char_count=char_count, chunk_count=chunk_count)


@dataclass
class ChatMessage:
    """One turn of the conversation, kept in Streamlit session state.

    ``sources`` holds the chunks that were used to build an assistant answer, so
    the UI can re-draw the page numbers and excerpts when the page re-runs.
    """

    role: str  # "user" or "assistant"
    content: str
    sources: list[RetrievedChunk] = field(default_factory=list)
