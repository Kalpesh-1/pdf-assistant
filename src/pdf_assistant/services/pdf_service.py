"""PDF inspection and page-by-page text extraction.

This is the first step of the RAG pipeline. Everything downstream - cleaning,
chunking, embedding, retrieval and finally the citation shown to the user -
depends on the page numbers produced here being correct.
"""

from __future__ import annotations

import pymupdf

from pdf_assistant.models.schemas import DocumentMetadata, PageText
from pdf_assistant.utils.file_validation import ValidationResult, validate_pdf_bytes
from pdf_assistant.utils.text_utils import clean_text


class PdfProcessingError(RuntimeError):
    """Raised when a PDF passed validation but could not be processed."""


def inspect_pdf(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int,
) -> tuple[ValidationResult, DocumentMetadata | None]:
    """Validate an upload and collect the document's basic facts.

    Args:
        file_bytes: The complete uploaded file.
        filename: Original filename of the upload.
        max_size_bytes: Upload limit from configuration.

    Returns:
        A tuple of the validation result and, when valid, the document metadata.
        ``char_count`` and ``chunk_count`` stay at zero until Phase 2 runs the
        extraction and chunking steps.
    """
    result = validate_pdf_bytes(file_bytes, filename, max_size_bytes)
    if not result.is_valid:
        return result, None

    metadata = DocumentMetadata(
        filename=filename,
        size_bytes=len(file_bytes),
        total_pages=result.total_pages,
    )
    return result, metadata


def extract_pages(file_bytes: bytes) -> list[PageText]:
    """Extract the cleaned text of every page, keeping the real page numbers.

    One :class:`PageText` is returned for each page in the document, in document
    order, *including* pages that turn out to be empty. Empty pages are kept for
    two reasons: the Document Details section lists every page, and silently
    dropping a page would make the position of a page in the list disagree with
    its printed page number - exactly the kind of mismatch that produces a wrong
    citation.

    Args:
        file_bytes: The complete PDF file, held in memory.

    Returns:
        A list of :class:`PageText`, ordered from page 1 to the last page.

    Raises:
        PdfProcessingError: If the document cannot be opened or a page cannot be
            read. PyMuPDF's own exception types are deliberately not allowed to
            escape, so callers only ever have to handle this one error.
    """
    pages: list[PageText] = []

    try:
        # stream= opens the PDF straight from memory, because the project never
        # writes an uploaded file to disk. filetype= stops PyMuPDF guessing.
        with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise PdfProcessingError(
                    "This PDF is password-protected, so its text cannot be read. "
                    "Please remove the password and upload it again."
                )

            # enumerate(..., start=1) does the 0-based -> 1-based conversion.
            #
            # WHY THIS MATTERS: PyMuPDF numbers pages from 0, but a reader looking
            # at the PDF in a viewer sees the first page as "page 1". Every
            # citation the app shows is built from this number, so if the
            # conversion were missing, every citation in the whole application
            # would be off by one. Converting here, once, means no later step has
            # to remember to adjust anything.
            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text()
                pages.append(PageText(page_number=page_number, text=clean_text(raw_text)))

    except PdfProcessingError:
        # Already a clear, user-facing message - pass it through untouched.
        raise
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several error types
        # The technical detail is included because it is genuinely helpful when
        # diagnosing a damaged file, and it exposes nothing sensitive.
        raise PdfProcessingError(
            f"The text of this PDF could not be read. The file may be damaged or "
            f"incomplete. Technical detail: {exc}"
        ) from exc

    return pages


def total_char_count(pages: list[PageText]) -> int:
    """Return the total number of characters across all extracted pages.

    Shown in the Overview and Document Details sections, and used as a quick
    sanity check that extraction actually produced text.
    """
    return sum(page.char_count for page in pages)
