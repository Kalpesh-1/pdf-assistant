"""Tests for session state handling.

These matter for correctness, not just tidiness: the specification requires that
replacing or deleting a PDF cannot leave anything behind from the previous
document.

``st.session_state`` normally needs a running Streamlit script, so these tests
swap in a plain dictionary. That is enough, because ``state.py`` only ever uses
session state as a mapping.
"""

from __future__ import annotations

import pytest

from pdf_assistant.models.schemas import ChatMessage, DocumentMetadata, ProcessingStatus
from pdf_assistant.ui import state


@pytest.fixture
def session(monkeypatch):
    """Replace st.session_state with a dictionary and return it."""
    fake_state: dict = {}
    monkeypatch.setattr(state.st, "session_state", fake_state)
    state.init_state()
    return fake_state


def test_init_state_creates_every_key(session):
    for key in state._DOCUMENT_STATE_DEFAULTS:
        assert key in session


def test_init_state_does_not_overwrite_existing_values(session):
    session["summary"] = "already here"

    state.init_state()

    assert session["summary"] == "already here"


def test_no_document_initially(session):
    assert state.has_document() is False
    assert state.is_ready() is False
    assert state.get_status() == ProcessingStatus.NO_DOCUMENT


def test_has_document_after_metadata_is_set(session):
    state.set_metadata(DocumentMetadata(filename="a.pdf", size_bytes=100, total_pages=2))

    assert state.has_document() is True


def test_clear_removes_all_document_derived_data(session):
    """The core safety requirement: nothing from the old document survives."""
    state.set_metadata(DocumentMetadata(filename="old.pdf", size_bytes=100, total_pages=2))
    state.set_status(ProcessingStatus.READY)
    session["pages"] = ["page content from the old document"]
    session["chunks"] = ["chunk from the old document"]
    session["vector_index"] = object()
    session["summary"] = "summary of the old document"
    state.add_chat_message(ChatMessage(role="user", content="question about the old document"))

    state.clear_document_state()

    assert state.get_metadata() is None
    assert session["pages"] == []
    assert session["chunks"] == []
    assert session["vector_index"] is None
    assert session["summary"] == ""
    assert state.get_chat_history() == []
    assert state.get_status() == ProcessingStatus.NO_DOCUMENT


def test_clear_removes_document_specific_widget_state(session):
    """Widget values must not carry over to the next document.

    A delete-confirmation box left ticked would let one click delete a document
    the user never confirmed, and a remembered page number could fall outside
    the next document's page range.

    The loop runs over ``_DOCUMENT_WIDGET_KEYS`` itself rather than naming the
    keys, so deleting an entry from that tuple cannot quietly pass this test -
    the key it protects would simply stop being checked.
    """
    assert state._DOCUMENT_WIDGET_KEYS, "there must be at least one document widget key"

    for widget_key in state._DOCUMENT_WIDGET_KEYS:
        session[widget_key] = "left over from the previous document"

    state.clear_document_state()

    for widget_key in state._DOCUMENT_WIDGET_KEYS:
        assert widget_key not in session


def test_the_known_document_widget_keys_are_still_registered(session):
    """The two widget keys that are known to leak must stay in the tuple.

    Paired with the generic test above: that one proves every registered key is
    cleared, this one proves the registry has not shrunk.
    """
    assert "details_confirm_delete" in state._DOCUMENT_WIDGET_KEYS
    assert "details_page_select" in state._DOCUMENT_WIDGET_KEYS


def test_current_explanation_key_is_plain_state_not_a_widget_value(session):
    """``current_explanation_key`` is written by code, so it is reset, not deleted.

    It is not a widget key: no widget uses it, application code assigns it. That
    means it belongs in _DOCUMENT_STATE_DEFAULTS, where clearing resets it to
    None rather than removing the key - so ``get_current_explanation_key`` can
    read it safely on the very next run.
    """
    assert "current_explanation_key" in state._DOCUMENT_STATE_DEFAULTS
    assert "current_explanation_key" not in state._DOCUMENT_WIDGET_KEYS

    state.set_current_explanation_key("about")
    assert state.get_current_explanation_key() == "about"

    state.clear_document_state()

    assert "current_explanation_key" in session
    assert state.get_current_explanation_key() is None


def test_clear_does_not_reset_the_uploader_round(session):
    """``upload_round`` must survive a clear, or deleting a document breaks.

    The round number is what forces st.file_uploader to forget the deleted file.
    The delete handler increments it and then clears the document state. If
    clearing also reset the counter, the uploader would go back to its previous
    key, hand the deleted file straight back, and re-process it. Keeping this
    key out of both tables is deliberate.
    """
    assert "upload_round" not in state._DOCUMENT_STATE_DEFAULTS
    assert "upload_round" not in state._DOCUMENT_WIDGET_KEYS

    session["upload_round"] = 3

    state.clear_document_state()

    assert session["upload_round"] == 3


def test_clear_chat_history_keeps_the_document_loaded(session):
    state.set_metadata(DocumentMetadata(filename="a.pdf", size_bytes=100, total_pages=1))
    state.add_chat_message(ChatMessage(role="user", content="hello"))

    state.clear_chat_history()

    assert state.get_chat_history() == []
    assert state.has_document() is True


def test_is_ready_only_when_status_is_ready(session):
    state.set_status(ProcessingStatus.EMBEDDING)
    assert state.is_ready() is False

    state.set_status(ProcessingStatus.READY)
    assert state.is_ready() is True


def test_failed_status_stores_the_error_message(session):
    state.set_status(ProcessingStatus.FAILED, "something went wrong")

    assert state.get_status() == ProcessingStatus.FAILED
    assert state.get_error_message() == "something went wrong"


def test_setting_a_non_failed_status_clears_the_old_error(session):
    state.set_status(ProcessingStatus.FAILED, "old error")

    state.set_status(ProcessingStatus.EXTRACTING)

    assert state.get_error_message() == ""


def test_upload_signature_detects_a_different_file(session):
    first = state.build_upload_signature("a.pdf", 1000)
    state.set_upload_signature(first)

    assert state.upload_changed(first) is False
    assert state.upload_changed(state.build_upload_signature("b.pdf", 1000)) is True


def test_upload_signature_detects_same_name_different_size(session):
    """An edited file reusing its name must still be treated as a new upload."""
    original = state.build_upload_signature("notes.pdf", 1000)
    state.set_upload_signature(original)

    assert state.upload_changed(state.build_upload_signature("notes.pdf", 2000)) is True


def test_chat_lists_are_not_shared_between_resets(session):
    """Guards against the classic mutable-default bug."""
    state.add_chat_message(ChatMessage(role="user", content="first"))
    state.clear_document_state()
    state.add_chat_message(ChatMessage(role="user", content="second"))

    assert [message.content for message in state.get_chat_history()] == ["second"]


def test_clear_removes_the_search_index(session):
    """The vector index is document-derived and must not survive a clear."""
    fake_index = object()
    state.set_search_index(fake_index)

    state.clear_document_state()

    assert state.get_search_index() is None


def test_clear_removes_the_embedding_model(session):
    """The loaded embedder is document-scoped and must be cleared."""
    fake_model = object()
    state.set_embedder(fake_model)

    state.clear_document_state()

    assert state.get_embedder() is None


def test_set_and_get_search_index_round_trip(session):
    """The index accessor must store and retrieve the index correctly."""
    fake_index = object()
    state.set_search_index(fake_index)

    assert state.get_search_index() is fake_index


def test_dict_defaults_are_copied_not_shared(session):
    """Guards against the mutable-default bug for dict-valued keys.

    A later task adds a dict-valued key. Without copying, the same dict object
    would be shared between documents - changes to one would leak into the next.
    This test ensures that when clear_document_state runs, dict defaults are
    copied just like list defaults already are.
    """
    # Add a dict to defaults temporarily to test the copy mechanism
    original_defaults = state._DOCUMENT_STATE_DEFAULTS.copy()
    state._DOCUMENT_STATE_DEFAULTS["test_dict_key"] = {"shared": "value"}

    try:
        # First document: modify the dict
        state.init_state()
        session["test_dict_key"]["shared"] = "modified"
        session["test_dict_key"]["new_key"] = "added"

        # Clear and check that the next document gets a fresh copy
        state.clear_document_state()

        assert session["test_dict_key"] == {"shared": "value"}
        assert "new_key" not in session["test_dict_key"]
    finally:
        # Clean up
        state._DOCUMENT_STATE_DEFAULTS = original_defaults
        session.pop("test_dict_key", None)


def test_clear_removes_cached_explanations(session):
    """Explanations are document-derived and must not survive a clear."""
    state.set_explanation("about", "This is about something")
    state.set_explanation("purpose", "The purpose is...")

    state.clear_document_state()

    assert state.get_explanation("about") == ""
    assert state.get_explanation("purpose") == ""


def test_explanation_cache_round_trip(session):
    """Explanation accessors must store and retrieve correctly."""
    state.set_explanation("about", "Document is about X")
    state.set_explanation("topics", "Topics: A, B, C")

    assert state.get_explanation("about") == "Document is about X"
    assert state.get_explanation("topics") == "Topics: A, B, C"
    assert state.get_explanation("nonexistent") == ""


def test_explanation_dicts_are_not_shared_between_documents(session):
    """Guards against the mutable-default bug for the explanations dict.

    Without copying the dict default in clear_document_state, the same dict
    object would be shared across documents and explanations from one document
    would leak into the next.
    """
    # First document: set some explanations
    state.set_explanation("about", "First document about")
    state.set_explanation("purpose", "First document purpose")

    # Clear and start a new document
    state.clear_document_state()

    # The new document should have an empty explanations dict
    assert state.get_explanation("about") == ""
    assert state.get_explanation("purpose") == ""

    # Set different explanations for the new document
    state.set_explanation("about", "Second document about")

    # Only the new value should be present
    assert state.get_explanation("about") == "Second document about"
    assert state.get_explanation("purpose") == ""


def test_summary_cache_round_trip(session):
    """Summary accessors must store and retrieve correctly."""
    state.set_summary("This is a summary")

    assert state.get_summary() == "This is a summary"
