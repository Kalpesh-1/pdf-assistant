"""Section 1: Overview.

Shows the uploaded document's basic facts, the processing status, and the
AI-generated overview (summary, important topics and key points).

The overview is generated on demand rather than automatically, because every
generation costs an API call on a rate-limited free tier. Once generated it is
cached in session state and cleared with the document.
"""

from __future__ import annotations

import streamlit as st

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import ProcessingStatus
from pdf_assistant.services.summary_service import explain_document
from pdf_assistant.ui import state
from pdf_assistant.ui.errors import GENERATION_ERRORS
from pdf_assistant.utils.text_utils import format_file_size

# The instruction for generating the overview content (summary, topics, key points).
# This asks for all three in one API call rather than making three separate requests,
# which saves on rate limits and keeps the response coherent.
OVERVIEW_INSTRUCTION = """Provide a clear overview of this document in three labelled \
markdown sections:

## Summary
A 4-6 sentence summary of what this document is about, in simple English.

## Important Topics
The main topics covered, as a bulleted list.

## Key Points
The most important facts and statements, as a bulleted list (one sentence each)."""


def render(settings: Settings) -> None:
    """Draw the Overview section."""
    st.header("Overview")

    if not state.has_document():
        st.info(
            "**No PDF uploaded yet.**\n\n"
            "Use the **Upload a PDF** control in the sidebar to get started. "
            "Once a document is loaded, its details and summary appear here."
        )
        _render_how_it_works()
        return

    metadata = state.get_metadata()
    assert metadata is not None  # guarded by has_document() above

    st.subheader(metadata.filename)

    # Four facts side by side, which reads better than a vertical list.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File size", format_file_size(metadata.size_bytes))
    col2.metric("Total pages", metadata.total_pages)
    col3.metric("Characters", f"{metadata.char_count:,}" if metadata.char_count else "-")
    col4.metric("Text chunks", metadata.chunk_count if metadata.chunk_count else "-")

    _render_status()
    _render_summary(settings)


def _render_status() -> None:
    """Show where the document is in the processing pipeline."""
    status = state.get_status()
    st.markdown("**Processing status**")

    if status == ProcessingStatus.READY:
        st.success(f"{status.value} - you can ask questions in the **Ask PDF** section.")
    elif status == ProcessingStatus.FAILED:
        st.error(f"{status.value}: {state.get_error_message()}")
    else:
        # Any mid-pipeline stage. Nothing to explain beyond where it has got to.
        st.info(status.value)


def _render_summary(settings: Settings) -> None:
    """Show the document summary, or explain why it is not available yet."""
    st.divider()
    st.markdown("**Document summary**")

    summary = state.get_summary()
    if summary:
        st.markdown(summary)
        # Show a regenerate button so the user can refresh the summary if needed
        if settings.has_api_key and state.is_ready():
            if st.button("Regenerate", key="overview_regenerate"):
                _generate_summary(settings)
        return

    if not settings.has_api_key:
        st.warning(
            "No Groq API key is configured, so the summary cannot be generated. "
            "Copy `.env.example` to `.env` and add your key."
        )
        return

    # Generation is explicit: the user clicks the button when they want it
    if not state.is_ready():
        if state.get_status() == ProcessingStatus.FAILED:
            st.info(
                "No summary can be generated because this document could not be "
                "processed. See the error above."
            )
        else:
            st.info("The summary becomes available once this document has finished processing.")
        return

    if st.button("Generate summary", key="overview_generate"):
        _generate_summary(settings)


def _generate_summary(settings: Settings) -> None:
    """Generate and cache the overview (summary, topics, key points).

    This is called when the user clicks Generate or Regenerate. It runs
    explain_document with OVERVIEW_INSTRUCTION to get all three sections in one
    API call, then caches the result so subsequent views are instant.
    """
    chunks = state.get_chunks()

    try:
        with st.spinner("Generating overview..."):
            # Get all three sections (summary, topics, key points) in one call
            overview = explain_document(settings, chunks, OVERVIEW_INSTRUCTION)
            state.set_summary(overview)
    except GENERATION_ERRORS as exc:
        st.error(f"Could not generate the overview: {exc}")
        return

    # Re-run so the new text is actually drawn.
    #
    # By the time this function is reached, _render_summary has already looked at
    # the (then empty) summary and drawn the button, so simply storing the text
    # would leave the screen unchanged: the user would watch the spinner finish
    # and see nothing. On "Regenerate" it is worse - the OLD text would still be
    # on screen after a successful regeneration, inviting a second billable
    # click. Re-running redraws the section from the freshly cached value.
    #
    # The rerun sits OUTSIDE the except block on purpose: on failure the script
    # must continue to the end so the st.error above stays visible.
    st.rerun()


def _render_how_it_works() -> None:
    """A short explanation shown on the empty state, useful during the viva."""
    st.divider()
    st.markdown("**How this application works**")
    st.markdown(
        """
        1. You upload a PDF, and the file is validated.
        2. The text is extracted page by page, keeping the real page numbers.
        3. The text is split into overlapping chunks.
        4. Each chunk is converted into a vector by a local embedding model.
        5. The vectors are stored in a FAISS index for fast similarity search.
        6. When you ask a question, the most relevant chunks are retrieved and
           sent to the language model, which answers using only that evidence.
        7. The answer is shown with the page numbers the evidence came from.
        """
    )
