"""Tests for upload validation.

These cover the error cases listed in the project specification: invalid files,
empty files, password-protected files, scanned files with no text, and files
over the size limit.
"""

from __future__ import annotations

from pdf_assistant.utils.file_validation import (
    validate_filename,
    validate_pdf_bytes,
    validate_size,
)

TWENTY_MB = 20 * 1024 * 1024


def test_valid_pdf_is_accepted(three_page_pdf):
    result = validate_pdf_bytes(three_page_pdf, "sample.pdf", TWENTY_MB)

    assert result.is_valid is True
    assert result.total_pages == 3
    assert result.message == ""


def test_page_count_is_reported_correctly(single_page_pdf):
    result = validate_pdf_bytes(single_page_pdf, "one.pdf", TWENTY_MB)

    assert result.is_valid is True
    assert result.total_pages == 1


def test_non_pdf_extension_is_rejected():
    result = validate_filename("notes.docx")

    assert result is not None
    assert result.is_valid is False
    assert "PDF" in result.message


def test_uppercase_pdf_extension_is_accepted():
    assert validate_filename("REPORT.PDF") is None


def test_empty_file_is_rejected():
    result = validate_pdf_bytes(b"", "empty.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert "empty" in result.message.lower()


def test_oversized_file_is_rejected():
    """A file above the limit must be refused before any parsing happens."""
    one_mb_limit = 1024 * 1024
    too_big = b"%PDF-" + b"x" * (one_mb_limit + 1)

    result = validate_pdf_bytes(too_big, "big.pdf", one_mb_limit)

    assert result.is_valid is False
    assert "larger than" in result.message


def test_size_within_limit_passes():
    assert validate_size(500, 1000) is None


def test_renamed_text_file_is_rejected(not_a_pdf):
    """A text file renamed to .pdf must be caught by the signature check."""
    result = validate_pdf_bytes(not_a_pdf, "fake.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert "does not look like a real PDF" in result.message


def test_password_protected_pdf_is_rejected(encrypted_pdf):
    result = validate_pdf_bytes(encrypted_pdf, "locked.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert "password" in result.message.lower()


def test_pdf_with_no_text_is_rejected_with_ocr_explanation(blank_pdf):
    """A scanned PDF must produce a clear message that OCR is not supported."""
    result = validate_pdf_bytes(blank_pdf, "scanned.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert "OCR" in result.message
    assert "scanned" in result.message.lower()


def test_corrupted_pdf_is_rejected():
    """A file with the right signature but broken content must not crash."""
    corrupted = b"%PDF-1.7\nthis content is not valid PDF structure at all"

    result = validate_pdf_bytes(corrupted, "broken.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert result.message != ""
