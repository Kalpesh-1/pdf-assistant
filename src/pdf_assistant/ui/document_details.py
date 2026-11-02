"""Section 4: Document Details.

A technical view of the loaded document: the file facts, the processing numbers,
which models are in use, and controls to inspect the extracted text or delete
the document.
"""

from __future__ import annotations

import streamlit as st

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import ProcessingStatus
from pdf_assistant.ui import state
from pdf_assistant.utils.text_utils import format_file_size


def render(settings: Settings) -> None:
    """Draw the Document Details section."""
    st.header("Document Details")

    if not state.has_document():
        st.info("**No PDF uploaded yet.** Upload a document in the sidebar to see its details.")
        return

    metadata = state.get_metadata()
    assert metadata is not None  # guarded by has_document() above

    index = state.get_search_index()

    # Build index status: show vector count when available
    if index is not None:
        vector_count = index.ntotal
        index_status = f"Built ({vector_count} vectors)"
    else:
        index_status = "Not built"

    # Format retrieval settings for display
    retrieval_settings = (
        f"Chunk size: {settings.chunk_size}, "
        f"Overlap: {settings.chunk_overlap}, "
        f"Top-k: {settings.top_k}, "
        f"Min similarity: {settings.min_similarity}"
    )

    # A single table is easier to read than a dozen separate labels, and it is
    # convenient to screenshot for the project report.
    st.table(
        {
            "Property": [
                "Original filename",
                "File size",
                "Total pages",
                "Extracted characters",
                "Number of text chunks",
                "Embedding model",
                "Language model",
                "Vector index status",
                "Retrieval settings",
                "Processing status",
            ],
            "Value": [
                metadata.filename,
                format_file_size(metadata.size_bytes),
                str(metadata.total_pages),
                f"{metadata.char_count:,}" if metadata.char_count else "Not extracted yet",
                str(metadata.chunk_count) if metadata.chunk_count else "Not created yet",
                settings.embedding_model,
                settings.groq_model,
                index_status,
                retrieval_settings,
                state.get_status().value,
            ],
        }
    )

    if state.get_status() == ProcessingStatus.FAILED:
        st.error(state.get_error_message())

    _render_extracted_text()
    _render_delete_control()


def _render_extracted_text() -> None:
    """Let the user inspect the text that was pulled out of the PDF.

    Being able to see the raw extracted text is genuinely useful: if an answer
    looks wrong, the usual cause is poor extraction rather than a bad model.
    """
    st.divider()
    st.markdown("**Extracted text**")

    pages = state.get_pages()
    if not pages:
        st.info("No extracted text available. Upload a PDF to see its text here.")
        return

    st.caption(
        "Useful for checking an answer: if a reply looks wrong, the cause is usually poor "
        "text extraction rather than the model."
    )

    with st.expander("View extracted text"):
        page_numbers = [page.page_number for page in pages]
        selected = st.selectbox("Page", page_numbers, key="details_page_select")
        page = next(page for page in pages if page.page_number == selected)

        if page.text:
            st.caption(f"{page.char_count:,} characters on page {page.page_number}")
            st.text_area(
                "Page text",
                value=page.text,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info(
                f"No text could be extracted from page {page.page_number}. This usually "
                "means the page is an image, such as a scan or a full-page diagram."
            )


def _render_delete_control() -> None:
    """Delete the document and everything derived from it.

    A confirmation checkbox guards the button, because losing the chat history
    by a stray click would be annoying.
    """
    st.divider()
    st.markdown("**Delete document**")
    st.caption(
        "Deleting removes the document, its extracted text, its chunks, the vector index "
        "and the chat history from this session."
    )

    confirmed = st.checkbox("Yes, I want to delete this document", key="details_confirm_delete")
    if st.button("Delete document", type="primary", disabled=not confirmed):
        # Bump the uploader's round number FIRST, before the state is cleared.
        #
        # st.file_uploader keeps the selected file under its own widget key, and
        # clearing document state cannot reach it. On the re-run below, the
        # sidebar would hand back the very same file, see that its signature no
        # longer matches (the signature was just cleared) and re-process it - so
        # the deleted document would come straight back, re-embedded, with its
        # chat history gone. Changing the key makes Streamlit build a brand-new,
        # empty uploader instead.
        previous_round = st.session_state.get("upload_round", 0)
        st.session_state["upload_round"] = previous_round + 1

        # Drop the retired uploader's own entry too. Nothing reads it any more,
        # but it still holds the raw bytes of the PDF the user just deleted, and
        # the rule for this project is that nothing from a removed document
        # stays reachable in session state.
        st.session_state.pop(f"pdf_upload_{previous_round}", None)

        # This also clears the confirmation box and the page selector, so the
        # next document starts from a clean slate.
        state.clear_document_state()
        st.success("Document deleted.")
        st.rerun()
