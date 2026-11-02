"""Shared pytest fixtures.

The PDF fixtures build real PDF files in memory with PyMuPDF rather than
committing binary sample files to the project. That keeps the test suite
self-contained and makes it obvious what each test document contains.
"""

from __future__ import annotations

import pymupdf
import pytest


class NetworkCallBlocked(RuntimeError):
    """Raised when a test tries to open a real connection to Groq.

    Derived from Exception, not BaseException, on purpose: a BaseException
    escaping the Streamlit script thread made AppTest behave differently from
    run to run. ``llm_service.complete`` therefore converts this into LlmError,
    which is harmless - the test is failed by the fixture teardown below, which
    is deterministic no matter what the application code does with the error.
    """


HOW_TO_FIX_A_BLOCKED_CALL = (
    "A test tried to construct a real Groq client, which would make a network "
    "call. Patch 'pdf_assistant.services.llm_service.Groq' with a mock, or "
    "patch the function the UI module actually looks up - for example "
    "'pdf_assistant.ui.overview.explain_document' rather than "
    "'pdf_assistant.services.summary_service.explain_document', because the UI "
    "module imported the function by name and holds its own reference to it."
)


@pytest.fixture(autouse=True)
def no_network_groq(monkeypatch):
    """Block real Groq API calls in every test.

    Every Groq call in this project goes through ``llm_service.complete``, which
    builds its client from the ``Groq`` name imported into that module.
    Replacing that one name is therefore enough to stop the whole suite reaching
    the network.

    This guard matters because a real call is invisible when it goes wrong: a
    fake key produces a 401, the service maps that to LlmError, and a test
    asserting only "an error was shown" passes for entirely the wrong reason -
    while making the suite depend on the internet.

    Applied automatically to every test. The tests that deliberately exercise
    the error mapping in ``llm_service`` patch the same name themselves with
    ``unittest.mock.patch``; that patch is applied after this fixture and undone
    when it exits, so those tests keep working and the guard is back in place
    for the next test.
    """
    attempts: list[str] = []

    class _ForbiddenGroqClient:
        """Stand-in for ``groq.Groq`` that refuses to be constructed."""

        def __init__(self, *args, **kwargs):
            attempts.append(HOW_TO_FIX_A_BLOCKED_CALL)
            raise NetworkCallBlocked(HOW_TO_FIX_A_BLOCKED_CALL)

    monkeypatch.setattr(
        "pdf_assistant.services.llm_service.Groq",
        _ForbiddenGroqClient,
    )

    yield

    # Fail the test here rather than relying on the raise above surfacing. The
    # application deliberately turns service errors into friendly messages, so
    # the raise alone could be swallowed and the test would still pass.
    if attempts:
        pytest.fail(attempts[0], pytrace=False)


def _build_pdf(pages: list[str], *, password: str | None = None) -> bytes:
    """Create a PDF in memory, one page per string in ``pages``."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        if text:
            # point=(72, 72) is one inch in from the top-left corner.
            page.insert_text((72, 72), text, fontsize=11)

    if password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    else:
        data = document.tobytes()

    document.close()
    return data


@pytest.fixture
def three_page_pdf() -> bytes:
    """A three-page PDF with distinct, identifiable text on each page.

    The marker words let a test confirm that text from page 2 really is reported
    as coming from page 2.
    """
    return _build_pdf(
        [
            "Page one discusses APPLE and the introduction.",
            "Page two discusses BANANA and the main method.",
            "Page three discusses CHERRY and the conclusion.",
        ]
    )


@pytest.fixture
def single_page_pdf() -> bytes:
    """A minimal valid one-page PDF with readable text."""
    return _build_pdf(["This document contains a reasonable amount of readable text."])


@pytest.fixture
def blank_pdf() -> bytes:
    """A PDF with two pages that contain no text at all.

    This stands in for a scanned document: the pages exist, but no text can be
    extracted from them.
    """
    return _build_pdf(["", ""])


@pytest.fixture
def encrypted_pdf() -> bytes:
    """A password-protected PDF."""
    return _build_pdf(["Secret content behind a password."], password="test-password")


@pytest.fixture
def not_a_pdf() -> bytes:
    """A plain text file pretending to be a PDF."""
    return b"This is just a text file, not a PDF document at all."
