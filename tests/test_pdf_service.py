"""Tests for the PDF service.

Covers inspection and metadata, plus page-by-page text extraction. The
page-number tests are the most important ones in the suite: every citation the
application shows is only as trustworthy as the page numbers produced here.

No test here touches the network or any API. The PDFs are built in memory by the
fixtures in ``conftest.py``.
"""

from __future__ import annotations

import pytest

from pdf_assistant.services.pdf_service import (
    PdfProcessingError,
    extract_pages,
    inspect_pdf,
    total_char_count,
)

TWENTY_MB = 20 * 1024 * 1024


def _page_number_containing(pages, marker: str) -> int:
    """Return the page number of the single page whose text contains ``marker``."""
    matches = [page.page_number for page in pages if marker in page.text]
    assert len(matches) == 1, f"expected {marker!r} on exactly one page, found {matches}"
    return matches[0]


def test_inspect_returns_metadata_for_a_valid_pdf(three_page_pdf):
    result, metadata = inspect_pdf(three_page_pdf, "report.pdf", TWENTY_MB)

    assert result.is_valid is True
    assert metadata is not None
    assert metadata.filename == "report.pdf"
    assert metadata.total_pages == 3
    assert metadata.size_bytes == len(three_page_pdf)


def test_inspect_reports_zero_counts_before_extraction(single_page_pdf):
    """Character and chunk counts stay at zero until Phase 2 fills them in."""
    _, metadata = inspect_pdf(single_page_pdf, "one.pdf", TWENTY_MB)

    assert metadata is not None
    assert metadata.char_count == 0
    assert metadata.chunk_count == 0


def test_inspect_returns_no_metadata_for_an_invalid_pdf(not_a_pdf):
    result, metadata = inspect_pdf(not_a_pdf, "fake.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert metadata is None


def test_inspect_returns_no_metadata_for_an_encrypted_pdf(encrypted_pdf):
    result, metadata = inspect_pdf(encrypted_pdf, "locked.pdf", TWENTY_MB)

    assert result.is_valid is False
    assert metadata is None
    assert "password" in result.message.lower()


def test_extract_returns_one_entry_per_page_in_order(three_page_pdf):
    pages = extract_pages(three_page_pdf)

    assert len(pages) == 3
    assert [page.page_number for page in pages] == [1, 2, 3]


def test_extract_numbers_pages_from_one_not_zero(three_page_pdf):
    """PyMuPDF counts pages from 0; a human reader counts from 1."""
    pages = extract_pages(three_page_pdf)

    assert pages[0].page_number == 1
    assert pages[-1].page_number == 3
    assert all(page.page_number > 0 for page in pages)


def test_extract_keeps_text_on_its_original_page(three_page_pdf):
    """The marker words must come back attached to the page they were written on.

    This is the behaviour every citation in the application depends on.
    """
    pages = extract_pages(three_page_pdf)

    assert _page_number_containing(pages, "APPLE") == 1
    assert _page_number_containing(pages, "BANANA") == 2
    assert _page_number_containing(pages, "CHERRY") == 3


def test_extract_cleans_the_text_it_returns(three_page_pdf):
    pages = extract_pages(three_page_pdf)

    for page in pages:
        assert "  " not in page.text, "doubled spaces should have been collapsed"
        assert page.text == page.text.strip(), "text should be stripped"
        assert "\t" not in page.text


def test_extract_reports_the_text_of_a_single_page_document(single_page_pdf):
    pages = extract_pages(single_page_pdf)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "readable text" in pages[0].text


def test_extract_keeps_empty_pages_so_numbering_stays_correct(blank_pdf):
    """A page with no text is still a page, and must still be listed."""
    pages = extract_pages(blank_pdf)

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["", ""]


def test_extract_raises_a_clear_error_for_a_file_that_is_not_a_pdf(not_a_pdf):
    with pytest.raises(PdfProcessingError) as error:
        extract_pages(not_a_pdf)

    assert "could not be read" in str(error.value)


def test_extract_raises_a_clear_error_for_a_corrupted_pdf():
    """Correct PDF signature, but the body is nonsense.

    Checking the magic bytes is not enough on its own, so extraction still has
    to survive a file that only *looks* like a PDF at the very start.
    """
    corrupted = b"%PDF-1.7\n" + b"\x00\x01\x02 this is not valid PDF structure" * 10

    with pytest.raises(PdfProcessingError):
        extract_pages(corrupted)


def test_extract_raises_a_clear_error_for_empty_input():
    with pytest.raises(PdfProcessingError):
        extract_pages(b"")


def test_extract_raises_a_clear_error_for_an_encrypted_pdf(encrypted_pdf):
    """Validation normally blocks this first, so extraction checks it too."""
    with pytest.raises(PdfProcessingError) as error:
        extract_pages(encrypted_pdf)

    assert "password" in str(error.value).lower()


def test_total_char_count_sums_every_page(three_page_pdf):
    pages = extract_pages(three_page_pdf)

    expected = len(pages[0].text) + len(pages[1].text) + len(pages[2].text)
    assert total_char_count(pages) == expected
    assert total_char_count(pages) > 0


def test_total_char_count_is_zero_for_a_blank_pdf(blank_pdf):
    assert total_char_count(extract_pages(blank_pdf)) == 0


def test_total_char_count_is_zero_for_no_pages():
    assert total_char_count([]) == 0
