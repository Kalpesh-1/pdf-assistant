"""Section 2: Ask PDF.

The chat interface. A question is answered by retrieving the most similar chunks
from the vector index and sending only those to the language model, so every
answer is grounded in the uploaded PDF.

Citations are built here from the retrieved chunks' stored page numbers. The
model is never asked for a page number, which is what makes a citation
trustworthy.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import ChatMessage, TextChunk
from pdf_assistant.prompts.qa_prompt import INSUFFICIENT_CONTEXT_MESSAGE
from pdf_assistant.services.llm_service import answer_question
from pdf_assistant.services.retrieval_service import has_sufficient_context, retrieve
from pdf_assistant.ui import state
from pdf_assistant.ui.errors import GENERATION_ERRORS
from pdf_assistant.utils.text_utils import build_excerpt


def render(settings: Settings) -> None:
    """Draw the Ask PDF section."""
    st.header("Ask PDF")

    if not state.has_document():
        st.info(
            "**No PDF uploaded yet.** Upload a document in the sidebar, then come back "
            "here to ask questions about it."
        )
        return

    if not settings.has_api_key:
        st.error(
            "**Groq API key missing.** Answer generation needs a key. "
            "Copy `.env.example` to `.env`, add your key from "
            "https://console.groq.com/keys, then restart the application."
        )
        return

    _render_toolbar()
    _render_history()
    _handle_new_question(settings)


def _render_toolbar() -> None:
    """Show the clear-chat control."""
    history = state.get_chat_history()
    left, right = st.columns([3, 1])
    left.caption(f"{len(history)} message(s) in this conversation.")
    if right.button("Clear chat", use_container_width=True, disabled=not history):
        state.clear_chat_history()
        st.rerun()


def _render_history() -> None:
    """Replay the stored conversation.

    Streamlit re-runs the script on every interaction, so previous messages have
    to be drawn again from session state each time rather than persisting on
    screen by themselves.
    """
    for message in state.get_chat_history():
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.sources:
                render_sources(message)


def render_sources(message: ChatMessage) -> None:
    """Show the page numbers and excerpts that support an answer.

    These come from the retrieved chunks' stored metadata, built in application
    code. The language model never supplies a page number, so a citation shown
    here always points at text that genuinely exists in the PDF.
    """
    pages = sorted({retrieved.chunk.page_number for retrieved in message.sources})
    page_label = ", ".join(str(page) for page in pages)

    with st.expander(f"Sources - page(s) {page_label}"):
        for retrieved in message.sources:
            st.markdown(
                f"**Page {retrieved.chunk.page_number}** (similarity score {retrieved.score:.3f})"
            )
            st.caption(build_excerpt(retrieved.chunk.text))


def build_assistant_reply(
    settings: Settings,
    question: str,
    chunks: list[TextChunk],
    index: Any,
    model: Any,
    history: list[ChatMessage],
) -> ChatMessage:
    """Generate an answer to a question using retrieval and the language model.

    This function extracts the decision logic from the chat handler so it can be
    unit-tested with fakes, which is far simpler than driving ``st.chat_input``
    through ``AppTest``.

    Args:
        settings: Configuration including API key, top_k, and min_similarity.
        question: The user's question.
        chunks: All chunks from the document, in embedding order.
        index: The FAISS search index.
        model: The loaded embedding model.
        history: Previous conversation turns. The caller must capture this BEFORE
            appending the current question to session state, so the question does
            not appear twice (once in history, once as the current question). As a
            safety net, this function checks if the last entry matches the question
            and strips it, but correct callers should not rely on that.

    Returns:
        A ChatMessage ready to append to the conversation, with role="assistant".
        On success, sources holds the retrieved chunks. On failure (abstention or
        error), sources is empty and content explains what went wrong.

    The flow is:
        1. Retrieve chunks matching the question.
        2. If context is insufficient, return the exact abstention message WITHOUT
           calling the language model (saves API quota).
        3. Otherwise call the model and return its answer with sources.
        4. On any error, return a readable error message as an assistant reply so
           the conversation stays coherent and the user can retry.
    """
    try:
        # Step 1: Retrieve relevant chunks
        retrieved = retrieve(
            question, chunks, index, model, settings.top_k, settings.min_similarity
        )

        # Step 2: Check if we have enough context to answer
        if not has_sufficient_context(retrieved):
            # Return the exact abstention message WITHOUT calling the model.
            # This is deliberate: it saves an API call on a rate-limited free tier
            # and guarantees the exact required wording.
            return ChatMessage(
                role="assistant",
                content=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
            )

        # Step 3: We have sufficient context - call the model
        # Defensive: if the caller accidentally passed history that includes the
        # current question as its last entry, strip it. This should not happen if
        # the UI correctly captures history BEFORE appending the new question, but
        # checking only the last position (not all messages with matching content)
        # prevents corrupting the conversation if a question is asked twice.
        clean_history = history
        if history and history[-1].role == "user" and history[-1].content == question:
            clean_history = history[:-1]

        answer = answer_question(
            settings=settings,
            question=question,
            retrieved=retrieved,
            history=clean_history,
        )

        return ChatMessage(
            role="assistant",
            content=answer,
            sources=retrieved,
        )

    except GENERATION_ERRORS as exc:
        # Any service error (LlmError, RetrievalError, etc.) becomes a readable
        # assistant message so the conversation stays coherent. The user can read
        # what went wrong and retry.
        return ChatMessage(
            role="assistant",
            content=str(exc),
            sources=[],
        )


def _handle_new_question(settings: Settings) -> None:
    """Accept a question and generate an answer using retrieval and the LLM.

    The decision logic is extracted as ``build_assistant_reply`` so it can be
    unit-tested without Streamlit. This function just wires up the UI widgets.
    """
    # Show disabled state with explanation when not ready
    if not state.is_ready():
        status = state.get_status()
        error_msg = state.get_error_message()
        if error_msg:
            st.warning(f"⚠️ Processing failed: {error_msg}")
        else:
            st.info(f"⏳ Document is being processed. Status: **{status.value}**")

    question = st.chat_input(
        "Ask a question about the uploaded PDF",
        disabled=not state.is_ready(),
    )
    if not question:
        return

    # CRITICAL: Capture history BEFORE appending the user's message.
    # get_chat_history() returns the live list from session state, so we must
    # take a snapshot before mutation to prevent the appended message from
    # retroactively appearing in what we pass to build_assistant_reply.
    history = list(state.get_chat_history())

    # Now append the user's message
    state.add_chat_message(ChatMessage(role="user", content=question))

    # If the document is not ready (this should not happen due to disabled input,
    # but defensive), explain why and return
    if not state.is_ready():
        status = state.get_status()
        error_msg = state.get_error_message()
        if error_msg:
            message = f"Questions cannot be answered yet. Processing failed: {error_msg}"
        else:
            message = f"Questions cannot be answered yet. Document status: {status.value}"

        state.add_chat_message(ChatMessage(role="assistant", content=message))
        st.rerun()
        return

    # Generate the reply under a spinner
    with st.spinner("Thinking..."):
        reply = build_assistant_reply(
            settings=settings,
            question=question,
            chunks=state.get_chunks(),
            index=state.get_search_index(),
            model=state.get_embedder(),
            history=history,
        )

    # Append the assistant's reply
    state.add_chat_message(reply)

    st.rerun()
