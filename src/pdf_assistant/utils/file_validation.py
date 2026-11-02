"""Upload validation.

Every uploaded file is checked here before any processing happens. Rejecting a
bad file early is much friendlier than letting it fail somewhere deep in the
pipeline with a confusing traceback.

The checks are deliberately ordered from cheapest to most expensive: extension
and size are simple arithmetic, while opening the PDF requires PyMuPDF to parse
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from pdf_assistant.utils.text_utils import format_file_size

# A real PDF file always starts with these bytes. Checking this catches a file
# that has simply been renamed to ".pdf".
PDF_MAGIC_BYTES = b"%PDF-"

# Below this many characters of extractable text, we treat the PDF as scanned
# (an image-only document) rather than a text document.
MIN_EXTRACTABLE_CHARS = 20


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one upload.

    Attributes:
        is_valid: True when the file can be processed.
        message: A plain-English explanation to show in the UI. Empty on success.
        total_pages: Page count, available only when validation succeeded.
    """

    is_valid: bool
    message: str = ""
    total_pages: int = 0

    @classmethod
    def ok(cls, total_pages: int) -> ValidationResult:
        return cls(is_valid=True, total_pages=total_pages)

    @classmethod
    def error(cls, message: str) -> ValidationResult:
        return cls(is_valid=False, message=message)


def validate_filename(filename: str) -> ValidationResult | None:
    """Check the file extension. Returns ``None`` when the name is acceptable."""
    if not filename or not filename.lower().endswith(".pdf"):
        return ValidationResult.error(
            "Only PDF files are supported. Please upload a file ending in .pdf."
        )
    return None


def validate_size(size_bytes: int, max_size_bytes: int) -> ValidationResult | None:
    """Check the upload size. Returns ``None`` when the size is acceptable."""
    if size_bytes <= 0:
        return ValidationResult.error("This file is empty (0 bytes). Please upload a valid PDF.")
    if size_bytes > max_size_bytes:
        return ValidationResult.error(
            f"This file is {format_file_size(size_bytes)}, which is larger than the "
            f"{format_file_size(max_size_bytes)} limit. Please upload a smaller PDF, "
            f"or raise MAX_UPLOAD_SIZE_MB in your .env file."
        )
    return None


def validate_pdf_bytes(file_bytes: bytes, filename: str, max_size_bytes: int) -> ValidationResult:
    """Run all upload checks on a PDF held in memory.

    The file is validated from bytes rather than from a path because Streamlit
    hands over an in-memory upload, and the project does not store PDFs on disk.

    Checks, in order:
        1. The name ends in ``.pdf``.
        2. The size is above zero and within the configured limit.
        3. The content really starts with the PDF signature.
        4. PyMuPDF can parse it.
        5. It is not password-protected.
        6. It has at least one page.
        7. At least some text can actually be extracted (i.e. not a scan).

    Args:
        file_bytes: The complete uploaded file.
        filename: Original name of the upload, used for the extension check.
        max_size_bytes: Upload limit from configuration.

    Returns:
        A :class:`ValidationResult`. On success it carries the page count, which
        saves the caller from opening the document a second time.
    """
    for check in (
        validate_filename(filename),
        validate_size(len(file_bytes), max_size_bytes),
    ):
        if check is not None:
            return check

    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        return ValidationResult.error(
            "This file does not look like a real PDF. It may be a different file type "
            "that has been renamed. Please upload a genuine PDF."
        )

    try:
        # stream= opens the PDF from memory. filetype= stops PyMuPDF guessing.
        with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
            if document.needs_pass:
                return ValidationResult.error(
                    "This PDF is password-protected. Please remove the password and "
                    "upload it again."
                )

            total_pages = document.page_count
            if total_pages == 0:
                return ValidationResult.error(
                    "This PDF has no pages. Please upload a document with content."
                )

            # Peek at the first few pages to decide whether there is any text at
            # all. Only a sample is checked, because opening every page of a
            # large PDF twice would be slow.
            sample_chars = 0
            for page_index in range(min(total_pages, 5)):
                sample_chars += len(document[page_index].get_text().strip())
                if sample_chars >= MIN_EXTRACTABLE_CHARS:
                    break

            if sample_chars < MIN_EXTRACTABLE_CHARS:
                return ValidationResult.error(
                    "No readable text could be found in this PDF. It is most likely a "
                    "scanned document made of page images. This version of PDF Assistant "
                    "does not support OCR (reading text from images), so please upload a "
                    "PDF that contains selectable text."
                )

    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several error types
        # The exception text is included because it is genuinely useful when
        # debugging a damaged file, and it contains no sensitive information.
        return ValidationResult.error(
            f"This PDF could not be opened. It may be damaged or incomplete. "
            f"Technical detail: {exc}"
        )

    return ValidationResult.ok(total_pages=total_pages)
