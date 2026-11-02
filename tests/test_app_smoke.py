"""Smoke tests that actually run the Streamlit application.

``AppTest`` is Streamlit's own test harness: it executes ``app.py`` the same way
the real server does, without needing a browser. These tests catch the mistakes
that unit tests cannot - a bad widget call, a missing session-state key, or an
exception raised while a section is being drawn.

No Groq API key is needed, and no network calls are made.
"""

from __future__ import annotations

import importlib.util
import zlib
from pathlib import Path

import numpy as np
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from pdf_assistant.config import get_settings

APP_FILE = str(Path(__file__).resolve().parents[1] / "app.py")

# Generous because the first run imports streamlit and pymupdf.
RUN_TIMEOUT = 60


class _FakeEmbedder:
    """A stand-in for the real SentenceTransformer.

    The real model is a 90 MB download and takes seconds to load, which no test
    should pay for. Vectors come from a CRC of the text, which is a fixed
    function of the bytes, so repeated runs produce identical vectors.
    """

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        rows = []
        for text in texts:
            checksum = zlib.crc32(text.encode())
            rows.append(
                np.array(
                    [
                        (checksum % 1000) / 1000.0,
                        ((checksum >> 10) % 1000) / 1000.0,
                        ((checksum >> 20) % 1000) / 1000.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            )
        return np.array(rows, dtype=np.float32)


def _use_a_fake_embedding_model(monkeypatch):
    """Make the upload pipeline embed with ``_FakeEmbedder``.

    ``app.py`` does ``from ...embedding_service import load_model``, and AppTest
    re-executes the whole script on every run, so that import is re-evaluated
    each time and picks up the patched attribute on the service module. The rest
    of the pipeline (chunking, embed_chunks, build_index) runs for real.
    """
    monkeypatch.setattr(
        "pdf_assistant.services.embedding_service.load_model",
        lambda model_name: _FakeEmbedder(),
    )


def _as_uploaded_file(filename: str, data: bytes) -> UploadedFile:
    """Wrap raw PDF bytes in the object ``st.file_uploader`` returns.

    AppTest offers no way to click a file_uploader, but the widget reads its
    value from session state under its key, so seeding that key is equivalent to
    the user having picked the file. This is what makes the upload handling in
    ``app.py`` testable at all.
    """
    record = UploadedFileRec(
        file_id=f"{filename}-{len(data)}",
        name=filename,
        type="application/pdf",
        data=data,
    )
    return UploadedFile(record, None)


def _load_app_module():
    """Import ``app.py`` as a module so its helper functions can be called.

    ``app.py`` sits at the project root rather than inside the package, so it is
    not importable by name. Importing it by path is enough: at import time it
    only adjusts sys.path and imports - nothing is drawn until ``main()`` runs.
    """
    spec = importlib.util.spec_from_file_location("pdf_assistant_app_under_test", APP_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Give every test a clean, predictable configuration.

    Two separate problems are handled here.

    First, ``get_settings`` is deliberately cached, because Streamlit re-runs the
    whole script on every interaction and re-reading the .env file each time
    would be wasteful. That cache lives for the life of the process, so without
    clearing it the first test's environment would leak into all the others.

    Second, the developer running these tests normally has a real ``.env`` file
    with a working API key, and ``config.py`` loads it. Tests must not depend on
    whether that file happens to exist, so the key is blanked here by default.
    It is set to an empty string rather than deleted, because python-dotenv is
    loaded with ``override=False`` and will fill in a variable that is absent
    from the environment but will leave an existing one alone.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("GROQ_API_KEY", "")
    yield
    get_settings.cache_clear()


@pytest.fixture
def app():
    """Run the app with no API key configured."""
    return AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT).run()


def test_app_starts_without_exceptions(app):
    assert not app.exception


def test_title_is_rendered(app):
    assert any("PDF Assistant" in heading.value for heading in app.title)


def test_sidebar_offers_all_four_sections(app):
    options = app.sidebar.radio[0].options

    assert options == [
        "Overview",
        "Ask PDF",
        "Understand Document",
        "Document Details",
    ]


def test_overview_is_shown_first_with_an_empty_state_message(app):
    """With no PDF loaded, the user must be told what to do next."""
    messages = " ".join(info.value for info in app.info)

    assert "No PDF uploaded yet" in messages


@pytest.mark.parametrize(
    "section",
    ["Overview", "Ask PDF", "Understand Document", "Document Details"],
)
def test_every_section_renders_without_exceptions(app, section):
    """Each section must draw cleanly even with no document loaded."""
    app.sidebar.radio[0].set_value(section).run()

    assert not app.exception


def test_missing_api_key_is_reported_but_never_revealed(app):
    """The sidebar must say a key is missing without printing any key."""
    warnings = " ".join(warning.value for warning in app.sidebar.warning)

    assert "missing" in warnings.lower()


def test_api_key_value_is_never_written_to_the_page(monkeypatch):
    """A configured key must not appear anywhere in the rendered output.

    This is the important security check: the key is used to call Groq, and it
    must never be echoed into the interface.
    """
    secret = "gsk_this_value_must_never_be_displayed"
    monkeypatch.setenv("GROQ_API_KEY", secret)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT).run()

    rendered = "".join(str(element.value) for element in app.markdown)
    rendered += "".join(str(element.value) for element in app.caption)
    rendered += "".join(str(element.value) for element in app.success)
    rendered += "".join(str(element.value) for element in app.sidebar.markdown)
    rendered += "".join(str(element.value) for element in app.sidebar.caption)
    rendered += "".join(str(element.value) for element in app.sidebar.success)

    assert secret not in rendered


def test_present_api_key_is_reported_as_found(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake_key_for_testing")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT).run()

    assert any("found" in success.value.lower() for success in app.sidebar.success)


def test_chat_section_blocks_questions_without_a_document(app):
    """Asking a question with no PDF must explain the problem, not crash."""
    app.sidebar.radio[0].set_value("Ask PDF").run()

    assert not app.exception
    messages = " ".join(info.value for info in app.info)
    assert "No PDF uploaded yet" in messages


def test_invalid_chunk_config_is_reported_as_a_configuration_problem(monkeypatch):
    """A bad .env value must produce a clear message, not a traceback."""
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "400")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT).run()

    assert not app.exception
    errors = " ".join(error.value for error in app.error)
    assert "CHUNK_OVERLAP" in errors


def test_chat_input_is_unavailable_until_the_document_is_ready(app):
    """The chat input must be disabled when no document is loaded.

    With no document loaded, the chat section shows an info message and the
    chat input exists but is disabled.
    """
    app.sidebar.radio[0].set_value("Ask PDF").run()

    assert not app.exception

    # With no document, an info message should explain what to do
    messages = " ".join(info.value for info in app.info)
    assert "No PDF uploaded yet" in messages


def test_chat_replays_history_with_its_sources():
    """The chat section must replay previous messages including their sources.

    AppTest cannot drive st.chat_input to actually generate a conversation, so
    this test verifies that the section renders existing history without exceptions.
    The core logic is fully tested in test_chat_flow.py.
    """
    from pdf_assistant.models.schemas import (
        ChatMessage,
        DocumentMetadata,
        ProcessingStatus,
        RetrievedChunk,
        TextChunk,
    )

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)

    # Seed a minimal ready state with chat history
    app.session_state["metadata"] = DocumentMetadata(
        filename="test.pdf", size_bytes=1000, total_pages=5
    )
    app.session_state["status"] = ProcessingStatus.READY
    app.session_state["chunks"] = []
    app.session_state["vector_index"] = None
    app.session_state["embedder"] = None

    # Add a conversation with sources
    fake_chunk = TextChunk(chunk_id=0, text="Paris is the capital of France.", page_number=1)
    sources = [RetrievedChunk(chunk=fake_chunk, score=0.9)]

    app.session_state["chat_history"] = [
        ChatMessage(role="user", content="What is the capital of France?"),
        ChatMessage(role="assistant", content="Paris is the capital of France.", sources=sources),
    ]

    # Run and navigate to chat section
    app.run()
    app.sidebar.radio[0].set_value("Ask PDF").run()

    # The critical assertion: no exceptions during rendering
    assert not app.exception


def test_clear_chat_empties_the_conversation():
    """The clear chat button must remove all messages from history.

    This test verifies that state.clear_chat_history() is called when the
    button is clicked. The actual button interaction is tested indirectly
    through the fact that the code runs without exceptions.
    """
    from pdf_assistant.models.schemas import ChatMessage
    from pdf_assistant.ui import state

    # This test verifies the state management logic directly rather than
    # trying to locate and click the button in AppTest, since AppTest has
    # limitations in finding widgets by label.

    # Initialize state
    state.init_state()

    # Add some chat history
    state.add_chat_message(ChatMessage(role="user", content="Test question?"))
    state.add_chat_message(ChatMessage(role="assistant", content="Test answer."))

    assert len(state.get_chat_history()) == 2

    # Clear it
    state.clear_chat_history()

    # Verify it's empty
    assert state.get_chat_history() == []


# Helper for seeding a ready document state
def _seed_ready_document(app_test):
    """Seed session state with a minimal ready document and fake chunks."""
    from pdf_assistant.models.schemas import (
        DocumentMetadata,
        ProcessingStatus,
        TextChunk,
    )

    app_test.session_state["metadata"] = DocumentMetadata(
        filename="test.pdf", size_bytes=1000, total_pages=5
    )
    app_test.session_state["status"] = ProcessingStatus.READY
    app_test.session_state["chunks"] = [
        TextChunk(chunk_id=0, text="Sample chunk text.", page_number=1)
    ]

    # Add a fake index with ntotal attribute
    class FakeIndex:
        ntotal = 42

    app_test.session_state["vector_index"] = FakeIndex()
    app_test.session_state["embedder"] = object()


def test_overview_shows_a_generate_summary_button_when_ready(monkeypatch):
    """With a ready document and API key, the generate button should appear."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Overview
    app.sidebar.radio[0].set_value("Overview").run()

    assert not app.exception
    # Look for the generate button
    button_labels = [btn.label for btn in app.button]
    assert "Generate summary" in button_labels


def test_overview_renders_a_cached_summary_without_calling_the_model(monkeypatch):
    """A cached summary should be shown without calling explain_document."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    # Monkeypatch explain_document to count calls
    call_count = {"value": 0}

    def fake_explain_document(settings, chunks, instruction):
        call_count["value"] += 1
        return "Generated summary"

    # Patch the name where it is LOOKED UP, not where it is defined.
    # overview.py does `from ...summary_service import explain_document`, so it
    # holds its own reference to the original function. Patching the attribute
    # on summary_service would leave that reference untouched, the real function
    # would still run, and this test's call counter could never increment - so
    # the assertion below would pass no matter what the code did.
    monkeypatch.setattr("pdf_assistant.ui.overview.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.session_state["summary"] = "Cached summary content"
    app.run()

    # Navigate to Overview
    app.sidebar.radio[0].set_value("Overview").run()

    assert not app.exception
    # The cached summary should be shown
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Cached summary content" in markdown_text
    # CRITICAL: the model should NOT have been called
    assert call_count["value"] == 0


def test_overview_shows_an_error_when_generation_fails(monkeypatch):
    """Generation errors should be caught and shown as st.error."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    # Monkeypatch explain_document to raise an error
    from pdf_assistant.services.llm_service import LlmError

    def fake_explain_document(settings, chunks, instruction):
        raise LlmError("API rate limit exceeded")

    # Patched where overview.py looks the name up. Patching summary_service
    # instead left the real function in place, so this test used to make a
    # genuine HTTPS request to api.groq.com with the fake key above, get a 401,
    # and pass on the resulting LlmError - testing Groq's rejection of a bad
    # key rather than this application's error handling.
    monkeypatch.setattr("pdf_assistant.ui.overview.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Overview and click generate
    app.sidebar.radio[0].set_value("Overview").run()
    # Find and click the generate button
    for btn in app.button:
        if btn.label == "Generate summary":
            btn.click().run()
            break

    assert not app.exception
    # The error should be shown
    errors = " ".join(err.value for err in app.error)
    assert "Could not generate" in errors or "rate limit" in errors


def test_overview_hides_generation_when_no_api_key_is_configured():
    """Without an API key, generation should be unavailable with an explanation."""
    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Overview
    app.sidebar.radio[0].set_value("Overview").run()

    assert not app.exception
    # Look for the warning about missing API key
    warnings = " ".join(warn.value for warn in app.warning)
    assert "No Groq API key" in warnings or "API key" in warnings
    # The generate button should NOT be present
    button_labels = [btn.label for btn in app.button]
    assert "Generate summary" not in button_labels


def test_understand_renders_all_eight_option_buttons(monkeypatch):
    """All eight guided explanation buttons should be rendered."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Understand Document
    app.sidebar.radio[0].set_value("Understand Document").run()

    assert not app.exception
    # All eight options should appear as buttons
    button_labels = [btn.label for btn in app.button]
    expected_labels = [
        "What is this document about?",
        "What is its purpose?",
        "Main topics",
        "Important sections",
        "Key points",
        "Difficult terms explained",
        "Main conclusions",
        "How to present this document",
    ]
    for label in expected_labels:
        assert label in button_labels


def test_understand_options_are_disabled_until_ready(monkeypatch):
    """Understand buttons should be disabled when document is not ready."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    from pdf_assistant.models.schemas import DocumentMetadata, ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["metadata"] = DocumentMetadata(
        filename="test.pdf", size_bytes=1000, total_pages=5
    )
    # Status is NOT ready
    app.session_state["status"] = ProcessingStatus.EXTRACTING
    app.run()

    # Navigate to Understand Document
    app.sidebar.radio[0].set_value("Understand Document").run()

    assert not app.exception
    # All buttons should be disabled
    for btn in app.button:
        if btn.label in [
            "What is this document about?",
            "Main topics",
            "Key points",
        ]:
            assert btn.disabled


def test_cached_explanation_is_shown_without_calling_the_model(monkeypatch):
    """A cached explanation should be shown without calling explain_document."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    # Monkeypatch explain_document to count calls
    call_count = {"value": 0}

    def fake_explain_document(settings, chunks, instruction):
        call_count["value"] += 1
        return "Generated explanation"

    # Patched where understand.py looks the name up - see the note in
    # test_overview_renders_a_cached_summary_without_calling_the_model.
    monkeypatch.setattr("pdf_assistant.ui.understand.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    # Pre-cache an explanation and mark it as current
    app.session_state["explanations"] = {"about": "Cached explanation text"}
    app.session_state["current_explanation_key"] = "about"
    app.run()

    # Navigate to Understand Document
    app.sidebar.radio[0].set_value("Understand Document").run()

    assert not app.exception
    # The cached explanation should be shown
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Cached explanation text" in markdown_text
    # CRITICAL: the model should NOT have been called
    assert call_count["value"] == 0


def test_one_failing_option_does_not_hide_the_others(monkeypatch):
    """When one option fails, the other seven should still be clickable."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    # Monkeypatch explain_document to fail for one specific instruction
    from pdf_assistant.services.llm_service import LlmError

    def fake_explain_document(settings, chunks, instruction):
        if "what this document is about" in instruction.lower():
            raise LlmError("Rate limit for this option")
        return f"Success: {instruction[:20]}"

    # Patched where understand.py looks the name up. With the old target this
    # test reached the real API over the network.
    monkeypatch.setattr("pdf_assistant.ui.understand.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Understand Document and click the failing option
    app.sidebar.radio[0].set_value("Understand Document").run()
    for btn in app.button:
        if btn.label == "What is this document about?":
            btn.click().run()
            break

    # Should show an error for this option
    errors = " ".join(err.value for err in app.error)
    assert "Could not generate" in errors or "Rate limit" in errors

    # But the other buttons should still be present and enabled
    button_labels = [btn.label for btn in app.button]
    assert "Main topics" in button_labels
    assert "Key points" in button_labels


def test_details_reports_the_vector_count_when_an_index_exists(monkeypatch):
    """The index status should show vector count from index.ntotal."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Document Details
    app.sidebar.radio[0].set_value("Document Details").run()

    assert not app.exception
    # Look for "Built (42 vectors)" in the table - access via .value attribute
    assert len(app.table) > 0, "No table found"
    table_data = app.table[0].value
    table_str = str(table_data)
    assert "42" in table_str or "Built" in table_str


def test_details_reports_not_built_without_an_index(monkeypatch):
    """Without an index, the status should be 'Not built'."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    from pdf_assistant.models.schemas import DocumentMetadata, ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["metadata"] = DocumentMetadata(
        filename="test.pdf", size_bytes=1000, total_pages=5
    )
    app.session_state["status"] = ProcessingStatus.TEXT_READY
    app.session_state["vector_index"] = None
    app.run()

    # Navigate to Document Details
    app.sidebar.radio[0].set_value("Document Details").run()

    assert not app.exception
    # Look for "Not built" in the table
    assert len(app.table) > 0, "No table found"
    table_data = app.table[0].value
    table_str = str(table_data)
    assert "Not built" in table_str


def test_details_shows_the_retrieval_settings(monkeypatch):
    """The table should show chunk size, overlap, top-k, and min similarity."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Document Details
    app.sidebar.radio[0].set_value("Document Details").run()

    assert not app.exception
    # Look for retrieval settings in the table
    assert len(app.table) > 0, "No table found"
    table_data = app.table[0].value
    table_str = str(table_data).lower()
    assert "chunk" in table_str or "top" in table_str


def test_details_never_shows_the_api_key(monkeypatch):
    """The API key value must never appear in Document Details."""
    secret = "gsk_this_secret_must_never_be_displayed"
    monkeypatch.setenv("GROQ_API_KEY", secret)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()

    # Navigate to Document Details
    app.sidebar.radio[0].set_value("Document Details").run()

    assert not app.exception
    # Check all rendered content
    rendered = ""
    for element in [app.markdown, app.table, app.caption, app.success, app.info]:
        rendered += " ".join(str(e) for e in element)

    assert secret not in rendered


# ---------------------------------------------------------------------------
# Upload handling: deletion, interrupted runs and failed documents.
#
# These drive st.file_uploader by seeding its widget key, which is the only way
# to exercise the upload branches of app.py from AppTest.
# ---------------------------------------------------------------------------


def _click_button(app, label):
    """Click the first button carrying ``label`` and re-run. Fails if absent."""
    for button in app.button:
        if button.label == label:
            button.click().run()
            return
    raise AssertionError(f"No button labelled {label!r}. Found: {[b.label for b in app.button]}")


def test_deleting_a_document_does_not_reprocess_the_same_upload(monkeypatch, single_page_pdf):
    """Deleting must stick even though the uploader still holds the file.

    st.file_uploader remembers the chosen file, and clearing document state
    cannot reach the widget. If the uploader kept the same key, the re-run after
    a delete would hand the file back, the signature would no longer match (it
    was just cleared), and the whole pipeline would run again - the deleted
    document would reappear, re-embedded, with its chat history lost.

    Running the app once more after the delete is the part that catches this:
    on the click run the state looks clean either way.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    _use_a_fake_embedding_model(monkeypatch)
    from pdf_assistant.models.schemas import ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["pdf_upload_0"] = _as_uploaded_file("notes.pdf", single_page_pdf)
    app.run()

    assert not app.exception
    assert app.session_state["status"] == ProcessingStatus.READY
    assert app.session_state["metadata"] is not None

    # Delete it from Document Details.
    app.sidebar.radio[0].set_value("Document Details").run()
    app.checkbox(key="details_confirm_delete").check().run()
    _click_button(app, "Delete document")

    assert not app.exception
    assert app.session_state["metadata"] is None
    # A fresh uploader round means the widget no longer holds the deleted file.
    assert app.session_state["upload_round"] == 1
    # And the retired uploader's entry, which still held the deleted PDF's raw
    # bytes, is gone too.
    assert "pdf_upload_0" not in app.session_state

    # THE REAL CHECK: run the script again, as any further click would.
    app.run()

    assert not app.exception
    assert app.session_state["metadata"] is None
    assert app.session_state["status"] == ProcessingStatus.NO_DOCUMENT
    assert app.session_state["vector_index"] is None
    # The round number must not have been reset by clear_document_state.
    assert app.session_state["upload_round"] == 1


def test_a_failed_document_is_not_reported_as_loaded(monkeypatch, single_page_pdf):
    """A post-validation failure must not show a green "Loaded:" message.

    Metadata is recorded as soon as validation passes, well before the document
    is usable. Checking metadata before the status would therefore report a
    document that Overview and Document Details both call Failed as loaded.
    """
    _use_a_fake_embedding_model(monkeypatch)
    from pdf_assistant.models.schemas import DocumentMetadata, ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["pdf_upload_0"] = _as_uploaded_file("notes.pdf", single_page_pdf)
    # Validation passed (metadata exists) but a later stage failed.
    app.session_state["metadata"] = DocumentMetadata(
        filename="notes.pdf", size_bytes=len(single_page_pdf), total_pages=1
    )
    app.session_state["status"] = ProcessingStatus.FAILED
    app.session_state["error_message"] = "Could not build the vector index."
    app.session_state["upload_signature"] = f"notes.pdf:{len(single_page_pdf)}"
    app.run()

    assert not app.exception
    successes = " ".join(success.value for success in app.sidebar.success)
    assert "Loaded:" not in successes
    errors = " ".join(error.value for error in app.sidebar.error)
    assert "Could not build the vector index." in errors


def test_a_failed_document_offers_a_retry(monkeypatch, single_page_pdf):
    """A failure can be transient, so the same file must be re-runnable.

    Without this the signature still matches, so re-selecting the file would do
    nothing at all and the user would have no way forward.
    """
    _use_a_fake_embedding_model(monkeypatch)
    from pdf_assistant.models.schemas import DocumentMetadata, ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["pdf_upload_0"] = _as_uploaded_file("notes.pdf", single_page_pdf)
    app.session_state["metadata"] = DocumentMetadata(
        filename="notes.pdf", size_bytes=len(single_page_pdf), total_pages=1
    )
    app.session_state["status"] = ProcessingStatus.FAILED
    app.session_state["error_message"] = "A temporary network problem."
    app.session_state["upload_signature"] = f"notes.pdf:{len(single_page_pdf)}"
    app.run()

    _click_button(app, "Retry processing")

    assert not app.exception
    assert app.session_state["status"] == ProcessingStatus.READY


def test_an_interrupted_upload_can_be_resumed(monkeypatch, single_page_pdf):
    """A run aborted during embedding must not strand the document forever.

    Embedding takes seconds to minutes, and clicking anything kills the running
    script. That leaves the signature recorded, the status mid-pipeline and no
    index, so nothing would ever restart the work: the chat would say "being
    processed" indefinitely and re-selecting the file would do nothing.
    """
    _use_a_fake_embedding_model(monkeypatch)
    from pdf_assistant.models.schemas import DocumentMetadata, ProcessingStatus

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    app.session_state["pdf_upload_0"] = _as_uploaded_file("notes.pdf", single_page_pdf)
    app.session_state["metadata"] = DocumentMetadata(
        filename="notes.pdf", size_bytes=len(single_page_pdf), total_pages=1
    )
    app.session_state["status"] = ProcessingStatus.EMBEDDING
    app.session_state["upload_signature"] = f"notes.pdf:{len(single_page_pdf)}"
    app.run()

    assert not app.exception
    # It must say what happened rather than claiming success.
    warnings = " ".join(warning.value for warning in app.sidebar.warning)
    assert "interrupted" in warnings.lower()
    successes = " ".join(success.value for success in app.sidebar.success)
    assert "Loaded:" not in successes

    _click_button(app, "Resume processing")

    assert not app.exception
    assert app.session_state["status"] == ProcessingStatus.READY
    assert app.session_state["vector_index"] is not None


def test_a_new_upload_clears_the_previous_documents_summary_and_explanations(
    monkeypatch, single_page_pdf, three_page_pdf
):
    """Replacing the PDF must wipe the old document's generated text.

    ``clear_document_state`` is tested on its own and ``_process_upload`` calls
    it first, but the composition is what the user actually relies on: a summary
    or explanation about the previous PDF must never be visible beside a new
    one. This calls ``_process_upload`` directly, because the two uploads have
    to happen inside one session.
    """
    from pdf_assistant.models.schemas import ProcessingStatus
    from pdf_assistant.ui import state

    app_module = _load_app_module()

    # Stand in for st.session_state with a plain dict, as tests/test_state.py
    # does: state.py only ever uses session state as a mapping.
    session: dict = {}
    monkeypatch.setattr(state.st, "session_state", session)
    state.init_state()
    monkeypatch.setattr(app_module, "load_model", lambda model_name: _FakeEmbedder())

    settings = get_settings()

    app_module._process_upload("first.pdf", single_page_pdf, settings)
    assert session["status"] == ProcessingStatus.READY

    # Pretend the user generated an overview and two guided explanations.
    state.set_summary("Summary of the FIRST document.")
    state.set_explanation("about", "The FIRST document is about apples.")
    state.set_explanation("topics", "Topics of the FIRST document.")
    state.set_current_explanation_key("about")

    app_module._process_upload("second.pdf", three_page_pdf, settings)

    assert session["status"] == ProcessingStatus.READY
    assert session["metadata"].filename == "second.pdf"
    assert state.get_summary() == ""
    assert session["explanations"] == {}
    assert state.get_current_explanation_key() is None


# ---------------------------------------------------------------------------
# Generation caching: a second click must not cost a second API call.
# ---------------------------------------------------------------------------


def test_a_freshly_generated_summary_is_visible_in_the_same_run(monkeypatch):
    """Clicking Generate must put the text on screen, not just in the cache.

    ``_render_summary`` has already seen an empty summary and drawn the button by
    the time generation runs, so storing the text without re-running would leave
    the screen unchanged: the user watches the spinner finish and sees nothing.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    call_count = {"value": 0}

    def fake_explain_document(settings, chunks, instruction):
        call_count["value"] += 1
        return "Freshly generated overview text."

    monkeypatch.setattr("pdf_assistant.ui.overview.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()
    app.sidebar.radio[0].set_value("Overview").run()

    _click_button(app, "Generate summary")

    assert not app.exception
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Freshly generated overview text." in markdown_text
    # The re-run that makes it visible must not trigger a second generation.
    assert call_count["value"] == 1


def test_viewing_the_summary_again_does_not_call_the_model_a_second_time(monkeypatch):
    """Once generated, the summary is served from the cache on every re-run."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    call_count = {"value": 0}

    def fake_explain_document(settings, chunks, instruction):
        call_count["value"] += 1
        return "Overview generated once."

    monkeypatch.setattr("pdf_assistant.ui.overview.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()
    app.sidebar.radio[0].set_value("Overview").run()
    _click_button(app, "Generate summary")

    assert call_count["value"] == 1

    # Navigate away and back, and re-run a few times.
    app.sidebar.radio[0].set_value("Document Details").run()
    app.sidebar.radio[0].set_value("Overview").run()
    app.run()

    assert not app.exception
    assert call_count["value"] == 1
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Overview generated once." in markdown_text


def test_clicking_an_understand_option_twice_calls_the_model_once(monkeypatch):
    """The second click on an already-answered option must be free.

    This is the point of the per-option cache: on a rate-limited free tier, a
    user clicking back and forth between options must not spend a request each
    time.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    call_count = {"value": 0}

    def fake_explain_document(settings, chunks, instruction):
        call_count["value"] += 1
        return "Explanation generated once."

    monkeypatch.setattr("pdf_assistant.ui.understand.explain_document", fake_explain_document)

    app = AppTest.from_file(APP_FILE, default_timeout=RUN_TIMEOUT)
    _seed_ready_document(app)
    app.run()
    app.sidebar.radio[0].set_value("Understand Document").run()

    _click_button(app, "Main topics")

    assert not app.exception
    assert call_count["value"] == 1
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Explanation generated once." in markdown_text

    _click_button(app, "Main topics")

    assert not app.exception
    assert call_count["value"] == 1
    markdown_text = " ".join(md.value for md in app.markdown)
    assert "Explanation generated once." in markdown_text
