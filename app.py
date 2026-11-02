"""PDF Assistant - AI-Powered PDF Question Answering System.

Entry point for the Streamlit application. Run it with:

    uv run streamlit run app.py

This file deliberately contains only presentation and wiring. All the real work
lives in ``src/pdf_assistant/``, which keeps the business logic testable without
Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Put src/ on the import path so the application runs straight from the project
# folder without needing to be installed first.
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pdf_assistant.config import ConfigError, Settings, get_settings  # noqa: E402
from pdf_assistant.models.schemas import ProcessingStatus  # noqa: E402
from pdf_assistant.services.chunking_service import chunk_pages  # noqa: E402
from pdf_assistant.services.embedding_service import embed_chunks, load_model  # noqa: E402
from pdf_assistant.services.pdf_service import (  # noqa: E402
    PdfProcessingError,
    extract_pages,
    inspect_pdf,
    total_char_count,
)
from pdf_assistant.services.vector_store_service import build_index  # noqa: E402
from pdf_assistant.ui import chat, document_details, overview, state, understand  # noqa: E402
from pdf_assistant.ui.errors import GENERATION_ERRORS  # noqa: E402
from pdf_assistant.utils.text_utils import format_file_size  # noqa: E402

SECTIONS = ("Overview", "Ask PDF", "Understand Document", "Document Details")

# Stages that mean "the pipeline started but never reached READY or FAILED".
#
# Streamlit aborts the running script the moment the user clicks anything else,
# and embedding can take minutes (the very first run downloads about 90 MB).
# An abort during those stages leaves the upload signature recorded but no
# index, so the ordinary "same file, nothing to do" path would hide the problem
# forever. Spotting these statuses lets us offer the work again instead.
IN_FLIGHT_STATUSES = (
    ProcessingStatus.VALIDATING,
    ProcessingStatus.EXTRACTING,
    ProcessingStatus.CHUNKING,
    ProcessingStatus.EMBEDDING,
    ProcessingStatus.INDEXING,
)


def main() -> None:
    """Configure the page, then draw the sidebar and the selected section."""
    st.set_page_config(
        page_title="PDF Assistant",
        page_icon="📄",
        layout="wide",
    )

    # Configuration is read before anything else. A bad .env value is reported
    # here rather than causing a confusing failure later.
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.title("PDF Assistant")
        st.error(f"**Configuration problem.** {exc}")
        st.info("Fix the value in your `.env` file, then reload this page.")
        return

    state.init_state()

    selected_section = _render_sidebar(settings)

    st.title("PDF Assistant")
    st.caption("Upload a PDF and ask questions about it. Answers cite the source pages.")

    if selected_section == "Overview":
        overview.render(settings)
    elif selected_section == "Ask PDF":
        chat.render(settings)
    elif selected_section == "Understand Document":
        understand.render(settings)
    else:
        document_details.render(settings)


def _render_sidebar(settings: Settings) -> str:
    """Draw the sidebar: navigation, upload control and configuration status.

    Returns:
        The name of the section the user has selected.
    """
    with st.sidebar:
        st.title("PDF Assistant")

        selected_section = st.radio("Section", SECTIONS, key="nav_section")

        st.divider()
        _render_upload(settings)

        st.divider()
        _render_config_status(settings)

    return selected_section


def _render_upload(settings: Settings) -> None:
    """Handle the file upload control.

    Streamlit hands back the same uploaded file on every script re-run, so the
    upload is only processed when its signature (name plus size) differs from
    the document already loaded. Without that check, the PDF would be
    re-validated on every single click.
    """
    st.markdown("**Upload a PDF**")
    st.caption(f"One PDF at a time, up to {settings.max_upload_size_mb} MB.")

    # The uploader's key carries a round number so that deleting a document can
    # bump it and get a *fresh* uploader. A file_uploader with a fixed key keeps
    # holding the chosen file across re-runs, so after a delete the same file
    # would be handed back and immediately re-processed - the document the user
    # just removed would reappear. See the note on "upload_round" in ui/state.py.
    round_no = st.session_state.setdefault("upload_round", 0)

    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"pdf_upload_{round_no}",
    )

    if uploaded is None:
        return

    file_bytes = uploaded.getvalue()
    signature = state.build_upload_signature(uploaded.name, len(file_bytes))

    if not state.upload_changed(signature):
        # This exact file has already been handled. Re-show the outcome, because
        # Streamlit clears the screen on every re-run - without this, a rejection
        # message would vanish as soon as the user clicked anything else, while
        # the rejected file was still sitting in the uploader.
        _render_previous_outcome(uploaded.name, file_bytes, settings)
        return

    _process_upload(uploaded.name, file_bytes, settings)


def _render_previous_outcome(filename: str, file_bytes: bytes, settings: Settings) -> None:
    """Re-show what happened to the file that is already in the uploader.

    Order matters here. The status is checked BEFORE the metadata, because
    metadata is recorded as soon as validation passes - long before the document
    is actually usable. A failure during extraction, chunking or embedding
    therefore leaves metadata in place, and testing metadata first would print a
    green "Loaded:" message for a document that Overview and Document Details
    both report as Failed.
    """
    status = state.get_status()

    if status in IN_FLIGHT_STATUSES:
        # The script was aborted part-way through processing, most likely because
        # the user clicked something during the slow embedding step. The file is
        # still selected but the signature already matches, so nothing would ever
        # restart the work. Offer it deliberately rather than guessing.
        st.warning("Processing was interrupted.")
        st.caption(f"Stopped at: {status.value}.")
        if st.button("Resume processing", key="upload_resume"):
            _process_upload(filename, file_bytes, settings)
        return

    if status == ProcessingStatus.FAILED:
        st.error(state.get_error_message())
        # Same retry affordance: a failure can be transient (a dropped download
        # of the embedding model, a rate limit), and re-selecting the same file
        # would not re-trigger processing because the signature still matches.
        if st.button("Retry processing", key="upload_resume"):
            _process_upload(filename, file_bytes, settings)
        return

    metadata = state.get_metadata()
    if metadata is not None:
        st.success(f"Loaded: {metadata.filename}")


def _process_upload(filename: str, file_bytes: bytes, settings: Settings) -> None:
    """Validate a newly uploaded PDF and load it into session state.

    The previous document is cleared FIRST. This guarantees that if validation
    of the new file fails, no part of the old document - text, chunks, index or
    chat history - is left behind to be queried.
    """
    state.clear_document_state()
    state.set_upload_signature(signature=state.build_upload_signature(filename, len(file_bytes)))
    state.set_status(ProcessingStatus.VALIDATING)

    with st.spinner("Validating PDF..."):
        result, metadata = inspect_pdf(
            file_bytes=file_bytes,
            filename=filename,
            max_size_bytes=settings.max_upload_size_bytes,
        )

    if not result.is_valid or metadata is None:
        state.set_status(ProcessingStatus.FAILED, result.message)
        st.error(result.message)
        return

    state.set_metadata(metadata)

    # ------------------------------------------------------------------
    # Extract the text, then split it into chunks.
    #
    # Both steps run here, immediately after a successful upload, so the
    # document is processed once rather than on every script re-run.
    # ------------------------------------------------------------------
    try:
        state.set_status(ProcessingStatus.EXTRACTING)
        with st.spinner("Extracting text from each page..."):
            pages = extract_pages(file_bytes)

        state.set_status(ProcessingStatus.CHUNKING)
        with st.spinner("Splitting the text into chunks..."):
            chunks = chunk_pages(
                pages,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
    except (PdfProcessingError, ValueError) as exc:
        # ValueError covers bad chunking settings coming from .env.
        message = str(exc)
        state.set_status(ProcessingStatus.FAILED, message)
        st.error(message)
        return

    char_count = total_char_count(pages)

    # A PDF can pass validation (it has some text on an early page) and still
    # produce nothing useful overall, for example a mostly-scanned document
    # with a single text header. Catch that here rather than letting the user
    # ask questions against an empty index.
    if not chunks:
        message = (
            "No usable text could be extracted from this PDF. It may be a scanned "
            "document made of page images, which this version cannot read because "
            "OCR is not supported."
        )
        state.set_status(ProcessingStatus.FAILED, message)
        st.error(message)
        return

    state.set_pages(pages)
    state.set_metadata(metadata.with_counts(char_count=char_count, chunk_count=len(chunks)))

    # ------------------------------------------------------------------
    # Embed the chunks and build the vector index.
    #
    # The chunks, embedder and index are stored ATOMICALLY (RULING 4) only
    # after both steps succeed. If embedding fails halfway, we must not
    # leave a chunks list with no index, or an index built from a previous
    # document. On failure, set FAILED with the message and leave the
    # extracted text in place so Document Details still shows it, but
    # ensure vector_index stays None.
    # ------------------------------------------------------------------
    try:
        state.set_status(ProcessingStatus.EMBEDDING)
        # The first run downloads ~90 MB from Hugging Face and may take a few
        # minutes on slow networks. Subsequent runs are instant (the model is
        # cached both on disk and in memory via lru_cache).
        with st.spinner(
            "Creating embeddings... (First run downloads the embedding model, "
            "about 90 MB, and may take a few minutes.)"
        ):
            embedder = load_model(settings.embedding_model)
            vectors = embed_chunks(embedder, chunks)

        state.set_status(ProcessingStatus.INDEXING)
        with st.spinner("Building vector index..."):
            index = build_index(vectors)

        # ATOMIC STORE: Only after both embedding and indexing succeed do we
        # store the chunks, embedder and index. This ensures that the index
        # rows always align with the chunk list (critical for correct
        # citations), and that a failed embedding does not leave stale state.
        state.set_chunks(chunks)
        state.set_embedder(embedder)
        state.set_search_index(index)
        state.set_status(ProcessingStatus.READY)

    except GENERATION_ERRORS as exc:
        # Any of: EmbeddingError, VectorStoreError, LlmError, RetrievalError,
        # SummaryError. The last three should not occur here, but catching the
        # whole tuple is simpler and safer than listing the subset.
        message = str(exc)
        state.set_status(ProcessingStatus.FAILED, message)
        st.error(message)
        return

    st.success(f"Loaded: {metadata.filename}")
    st.caption(
        f"{metadata.total_pages} page(s), {format_file_size(metadata.size_bytes)}, "
        f"{char_count:,} characters, {len(chunks)} chunks."
    )


def _render_config_status(settings: Settings) -> None:
    """Show the active configuration.

    The API key is reported only as present or missing. Its value is never
    displayed, logged or written anywhere.
    """
    st.markdown("**Configuration**")

    if settings.has_api_key:
        st.success("Groq API key: found")
    else:
        st.warning("Groq API key: missing")
        st.caption("Copy `.env.example` to `.env` and add your key to enable AI features.")

    st.caption(f"Language model: `{settings.groq_model}`")
    st.caption(f"Embedding model: `{settings.embedding_model}`")
    st.caption(
        f"Chunk size: {settings.chunk_size} chars, "
        f"overlap: {settings.chunk_overlap}, top-k: {settings.top_k}"
    )


if __name__ == "__main__":
    main()
