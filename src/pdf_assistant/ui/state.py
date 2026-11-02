"""Streamlit session state management.

Streamlit re-runs the entire script every time the user clicks anything, so any
value that must survive between clicks lives in ``st.session_state``. This
module is the single gatekeeper for that state.

The important rule enforced here: when a document is cleared or replaced,
EVERYTHING derived from the old document goes with it - metadata, extracted
text, chunks, the vector index and the chat history. If any of those survived,
the application could answer a question about a PDF the user had already
removed, which would be a serious correctness and privacy bug.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from pdf_assistant.models.schemas import (
    ChatMessage,
    DocumentMetadata,
    PageText,
    ProcessingStatus,
    TextChunk,
)

# Every key the application stores, with the value it resets to. Keeping them in
# one dictionary means "clear the document" cannot accidentally miss one.
_DOCUMENT_STATE_DEFAULTS: dict[str, Any] = {
    "metadata": None,  # DocumentMetadata | None
    "status": ProcessingStatus.NO_DOCUMENT,
    "error_message": "",
    "pages": [],  # list[PageText]   - filled in Phase 2
    "chunks": [],  # list[TextChunk]  - filled in Phase 2
    "vector_index": None,  # FAISS index      - filled in Phase 3
    "embedder": None,  # SentenceTransformer | None - loaded model, filled in Phase 3/4
    "summary": "",  # document summary - filled in Phase 3
    "explanations": {},  # dict[str, str] - guided explanations cache - filled in Phase 4
    # Which of the eight guided explanations the user last asked to see. This is
    # written by application code, not by a widget, so it belongs here rather
    # than in _DOCUMENT_WIDGET_KEYS (a widget value would be *deleted* on clear;
    # this one is *reset* to None).
    "current_explanation_key": None,  # str | None
    "chat_history": [],  # list[ChatMessage]
    # Identifies the current upload so we can detect when the user picks a
    # different file and reset automatically.
    "upload_signature": None,
}

# DELIBERATELY NOT DOCUMENT STATE: "upload_round".
#
# This counter is part of the fix for "delete does not stick". st.file_uploader
# remembers the file the user picked, and that memory lives under the widget's
# own key. Deleting the document cannot clear the widget, so the only way to
# make Streamlit forget the file is to draw a *different* widget - which means
# giving the uploader a new key. "upload_round" is that key's version number and
# is bumped by the delete handler.
#
# It must therefore NOT appear in _DOCUMENT_STATE_DEFAULTS or
# _DOCUMENT_WIDGET_KEYS: clearing the document also clears those, which would
# reset the counter back to the value the just-deleted file is already attached
# to, re-arming exactly the bug it exists to prevent. Its absence here is
# intentional, not an omission.

# Keys belonging to widgets whose value is tied to the current document.
#
# Streamlit keeps a widget's value in session state under its key, and that
# value outlives the widget itself. These must be removed along with the
# document, for two reasons:
#   - "details_confirm_delete": a confirmation box left ticked for the previous
#     document would still be ticked for the next one, so a single click could
#     delete a document the user never confirmed.
#   - "details_page_select": a remembered page number such as 10 would be
#     restored into a page list that only goes up to 2 after a shorter PDF is
#     uploaded, which Streamlit cannot do.
_DOCUMENT_WIDGET_KEYS = ("details_confirm_delete", "details_page_select")


def init_state() -> None:
    """Create any missing state keys. Safe to call on every script run."""
    for key, default in _DOCUMENT_STATE_DEFAULTS.items():
        if key not in st.session_state:
            # Lists and dicts are copied so that separate keys never end up
            # sharing the same object (the classic mutable-default bug).
            if isinstance(default, list):
                st.session_state[key] = list(default)
            elif isinstance(default, dict):
                st.session_state[key] = dict(default)
            else:
                st.session_state[key] = default


def clear_document_state() -> None:
    """Remove the current document and everything derived from it.

    Called when the user deletes the document and when a different file is
    uploaded.
    """
    for key, default in _DOCUMENT_STATE_DEFAULTS.items():
        # Lists and dicts are copied so that separate keys never end up
        # sharing the same object. Without this, a dict default added by a
        # later task would be shared across documents, and changes to one
        # document's dict would leak into the next.
        if isinstance(default, list):
            st.session_state[key] = list(default)
        elif isinstance(default, dict):
            st.session_state[key] = dict(default)
        else:
            st.session_state[key] = default

    # Widget values are deleted rather than reset, so each widget starts from
    # its own default the next time it is drawn.
    for widget_key in _DOCUMENT_WIDGET_KEYS:
        st.session_state.pop(widget_key, None)


def clear_chat_history() -> None:
    """Empty the conversation but keep the document loaded."""
    st.session_state["chat_history"] = []


def has_document() -> bool:
    """True when a validated document is loaded."""
    return st.session_state.get("metadata") is not None


def is_ready() -> bool:
    """True when the document is fully processed and questions can be answered."""
    return st.session_state.get("status") == ProcessingStatus.READY


def get_metadata() -> DocumentMetadata | None:
    return st.session_state.get("metadata")


def set_metadata(metadata: DocumentMetadata) -> None:
    st.session_state["metadata"] = metadata


def get_pages() -> list[PageText]:
    """Return the extracted page texts, in document order."""
    return st.session_state.get("pages", [])


def set_pages(pages: list[PageText]) -> None:
    st.session_state["pages"] = pages


def get_chunks() -> list[TextChunk]:
    """Return the chunks built from the extracted pages."""
    return st.session_state.get("chunks", [])


def set_chunks(chunks: list[TextChunk]) -> None:
    st.session_state["chunks"] = chunks


def get_status() -> ProcessingStatus:
    return st.session_state.get("status", ProcessingStatus.NO_DOCUMENT)


def set_status(status: ProcessingStatus, error_message: str = "") -> None:
    """Record the pipeline stage, and an error message when it failed."""
    st.session_state["status"] = status
    st.session_state["error_message"] = error_message


def get_error_message() -> str:
    return st.session_state.get("error_message", "")


def get_chat_history() -> list[ChatMessage]:
    return st.session_state.get("chat_history", [])


def add_chat_message(message: ChatMessage) -> None:
    st.session_state.setdefault("chat_history", []).append(message)


def upload_changed(signature: str) -> bool:
    """True when ``signature`` describes a different file from the loaded one.

    The signature combines filename and size. Comparing it lets the application
    tell "the user uploaded a new file" apart from "Streamlit re-ran the script
    and handed back the same file again", which happens constantly.
    """
    return st.session_state.get("upload_signature") != signature


def set_upload_signature(signature: str) -> None:
    st.session_state["upload_signature"] = signature


def build_upload_signature(filename: str, size_bytes: int) -> str:
    """Build the identifier used to detect a changed upload."""
    return f"{filename}:{size_bytes}"


def get_search_index():
    """Return the FAISS vector index, or None if not yet built.

    The index is stored under the existing ``vector_index`` key in session state.
    This accessor exists to keep the key name encapsulated and to provide a
    typed interface for the UI to retrieve the index.
    """
    return st.session_state.get("vector_index")


def set_search_index(index) -> None:
    """Store the FAISS vector index in session state.

    The index is built after embedding succeeds and must be stored atomically
    with the chunks list - see RULING 4 in the task brief. If the index rows are
    out of order relative to the chunks, citations will point at the wrong pages.
    """
    st.session_state["vector_index"] = index


def get_embedder():
    """Return the loaded embedding model, or None if not yet loaded.

    The model is cached in session state (in addition to being lru_cached by the
    service) so the UI can pass the same instance to retrieval without
    re-resolving it. The key name is ``embedder`` rather than ``embedding_model``
    because Settings.embedding_model holds the model's *name string*, while this
    holds the *loaded object* - two different things with similar names would be
    exactly what a viva examiner probes.
    """
    return st.session_state.get("embedder")


def set_embedder(model) -> None:
    """Store the loaded embedding model in session state.

    The model object is held here after the first load so subsequent retrievals
    can reuse it without calling load_model again. The lru_cache in the service
    also caches it, but this session-state copy lets the UI explicitly track
    that the model has been loaded for the current document.
    """
    st.session_state["embedder"] = model


def get_summary() -> str:
    """Return the cached document summary, or empty string if not yet generated."""
    return st.session_state.get("summary", "")


def set_summary(text: str) -> None:
    """Cache the document summary in session state.

    Summaries are expensive (they process the whole document), so they are
    cached rather than regenerated on every page refresh. The cache is cleared
    along with other document state when the PDF changes.
    """
    st.session_state["summary"] = text


def get_explanation(key: str) -> str:
    """Return the cached explanation for a given option key, or empty if not run.

    Args:
        key: The option key (e.g., "about", "purpose", "topics").

    Returns:
        The cached explanation text, or empty string if this option has never
        been run for the current document.
    """
    explanations = st.session_state.get("explanations", {})
    return explanations.get(key, "")


def set_explanation(key: str, text: str) -> None:
    """Cache an explanation for a given option key.

    Each of the eight guided explanation options is cached separately, so
    re-selecting an already-run option is instant (no API call). The cache is
    cleared when the document changes.

    Args:
        key: The option key (e.g., "about", "purpose", "topics").
        text: The generated explanation text.
    """
    explanations = st.session_state.setdefault("explanations", {})
    explanations[key] = text


def get_current_explanation_key() -> str | None:
    """Return the option key of the explanation currently on display, if any.

    Only one of the eight guided explanations is shown at a time: the last one
    the user asked for. This records which, so the choice survives the script
    re-run that follows the click.
    """
    return st.session_state.get("current_explanation_key")


def set_current_explanation_key(key: str) -> None:
    """Mark an explanation as the one to display.

    Goes through this accessor rather than writing session state from the UI,
    because this module is meant to be the single gatekeeper for every
    document-scoped key - that is what makes "clear the document" auditable.
    """
    st.session_state["current_explanation_key"] = key
