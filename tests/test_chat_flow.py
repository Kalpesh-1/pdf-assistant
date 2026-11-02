"""Tests for the chat decision logic extracted as a plain function.

These test the core answer-generation flow without involving Streamlit widgets,
which cannot be driven by ``st.chat_input`` in ``AppTest``. The decision logic is
extracted as ``build_assistant_reply`` so it can be unit-tested with fakes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import ChatMessage, RetrievedChunk, TextChunk
from pdf_assistant.prompts.qa_prompt import INSUFFICIENT_CONTEXT_MESSAGE
from pdf_assistant.services.embedding_service import EmbeddingError
from pdf_assistant.services.llm_service import LlmError
from pdf_assistant.services.retrieval_service import RetrievalError
from pdf_assistant.services.summary_service import SummaryError
from pdf_assistant.services.vector_store_service import VectorStoreError
from pdf_assistant.ui.chat import build_assistant_reply


@pytest.fixture
def settings():
    """Minimal settings with a fake API key."""
    return Settings(
        groq_api_key="gsk_fake_key_for_testing",
        groq_model="test-model",
        embedding_model="test-embed",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )


@pytest.fixture
def fake_chunks():
    """Sample chunks for testing retrieval."""
    return [
        TextChunk(chunk_id=0, text="The capital of France is Paris.", page_number=1),
        TextChunk(chunk_id=1, text="Paris is known for the Eiffel Tower.", page_number=1),
        TextChunk(chunk_id=2, text="France is in Europe.", page_number=2),
    ]


@pytest.fixture
def fake_index():
    """A mock FAISS index."""
    index = MagicMock()
    index.ntotal = 3
    return index


@pytest.fixture
def fake_model():
    """A mock embedding model."""
    return MagicMock()


def test_reply_is_the_exact_abstention_string_when_context_is_weak(
    settings, fake_chunks, fake_index, fake_model
):
    """When no chunks clear the similarity floor, the reply must be the exact abstention."""
    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
    ):
        # Simulate retrieval finding nothing relevant
        mock_retrieve.return_value = []
        mock_has_sufficient.return_value = False

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    # The reply content must be EXACTLY the required string, character for character
    assert reply.content == INSUFFICIENT_CONTEXT_MESSAGE
    assert reply.role == "assistant"


def test_abstention_path_never_calls_the_model(settings, fake_chunks, fake_index, fake_model):
    """When context is insufficient, the LLM must not be called (saves API quota)."""
    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = []
        mock_has_sufficient.return_value = False

        build_assistant_reply(settings, "Where is Paris?", fake_chunks, fake_index, fake_model, [])

    # The model must never be called on this path
    mock_answer.assert_not_called()


def test_abstention_reply_has_no_sources(settings, fake_chunks, fake_index, fake_model):
    """The abstention reply must have an empty sources list."""
    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
    ):
        mock_retrieve.return_value = []
        mock_has_sufficient.return_value = False

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.sources == []


def test_reply_carries_the_retrieved_chunks_as_sources(
    settings, fake_chunks, fake_index, fake_model
):
    """When the model answers successfully, the reply must carry the retrieved chunks."""
    retrieved = [
        RetrievedChunk(chunk=fake_chunks[0], score=0.9),
        RetrievedChunk(chunk=fake_chunks[1], score=0.8),
    ]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.return_value = "Paris is the capital of France."

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.sources == retrieved


def test_reply_sources_hold_the_original_page_numbers(
    settings, fake_chunks, fake_index, fake_model
):
    """Citations must come from chunk metadata, never from model output."""
    retrieved = [
        RetrievedChunk(chunk=fake_chunks[0], score=0.9),
        RetrievedChunk(chunk=fake_chunks[2], score=0.7),
    ]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.return_value = "Paris is in France, which is in Europe."

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    # Extract page numbers from the reply's sources
    pages = [r.chunk.page_number for r in reply.sources]
    assert pages == [1, 2]


@pytest.mark.parametrize(
    "error_class,error_message",
    [
        (LlmError, "Rate limit reached. Please try again."),
        (RetrievalError, "The vector index holds 5 vectors but 3 chunks were supplied."),
        (EmbeddingError, "Failed to embed the question."),
        (VectorStoreError, "FAISS index search failed."),
        (SummaryError, "Failed to generate document summary."),
    ],
)
def test_all_generation_errors_become_readable_assistant_messages(
    settings, fake_chunks, fake_index, fake_model, error_class, error_message
):
    """All GENERATION_ERRORS must be caught and turned into readable assistant messages.

    This parametrized test covers all five error types that can be raised by
    services: LlmError, RetrievalError, EmbeddingError, VectorStoreError, and
    SummaryError. Each must produce an assistant message with the error text and
    empty sources.
    """
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True

        # Inject the error at the appropriate point
        if error_class == RetrievalError:
            mock_retrieve.side_effect = error_class(error_message)
        else:
            mock_answer.side_effect = error_class(error_message)

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.role == "assistant"
    assert error_message in reply.content
    assert reply.sources == []


def test_llm_error_becomes_a_readable_assistant_message(
    settings, fake_chunks, fake_index, fake_model
):
    """When the LLM fails, the error message must appear as a coherent assistant reply."""
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.side_effect = LlmError("Rate limit reached. Please try again.")

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.role == "assistant"
    assert "Rate limit reached" in reply.content


def test_llm_error_reply_has_no_sources(settings, fake_chunks, fake_index, fake_model):
    """Error replies must have empty sources, not leftover retrieval results."""
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.side_effect = LlmError("API timeout.")

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.sources == []


def test_api_key_never_appears_in_an_error_reply(settings, fake_chunks, fake_index, fake_model):
    """The API key must never leak into any error message shown to the user.

    This test verifies two things:
    1. The specific secret key value never appears
    2. No Groq API key prefix (gsk_) appears in realistic error scenarios

    The llm_service module carefully constructs error messages to avoid including
    the API key. This test ensures that pattern holds through the chat layer.
    """
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    # Use a recognizable fake key
    settings_with_secret = Settings(
        groq_api_key="gsk_secret_that_must_not_leak",
        groq_model="test-model",
        embedding_model="test-embed",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        # Simulate a realistic error message that does NOT contain the key
        # (llm_service carefully avoids including keys in error messages)
        mock_answer.side_effect = LlmError("Authentication failed. Your GROQ_API_KEY was rejected.")

        reply = build_assistant_reply(
            settings_with_secret, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    # The error message must not contain the secret key
    assert "gsk_secret_that_must_not_leak" not in reply.content

    # Additionally, no Groq API key prefix should appear anywhere in the reply.
    # This ensures that even partial key leaks are caught.
    assert "gsk_" not in reply.content


def test_history_is_passed_to_the_model_for_follow_ups(
    settings, fake_chunks, fake_index, fake_model
):
    """Follow-up questions must see previous conversation turns."""
    history = [
        ChatMessage(role="user", content="What is the capital of France?"),
        ChatMessage(role="assistant", content="Paris is the capital of France."),
    ]
    retrieved = [RetrievedChunk(chunk=fake_chunks[1], score=0.85)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.return_value = "It is known for the Eiffel Tower."

        build_assistant_reply(
            settings, "What is it known for?", fake_chunks, fake_index, fake_model, history
        )

    # answer_question must have been called with the history
    mock_answer.assert_called_once()
    call_args = mock_answer.call_args
    passed_history = call_args[1]["history"]
    assert passed_history == history


def test_current_question_is_not_duplicated_into_history(
    settings, fake_chunks, fake_index, fake_model
):
    """The current question must not appear inside the history passed to the model.

    The UI appends the user's message to session state before calling
    build_assistant_reply. If the function then passes the full session state as
    history, the question would be sent twice: once in history, once as the current
    question. This test ensures that does not happen.
    """
    # Simulate the state AFTER the UI has appended the new user message
    history_with_new_question = [
        ChatMessage(role="user", content="What is France?"),
        ChatMessage(role="assistant", content="France is a country in Europe."),
        ChatMessage(role="user", content="What is its capital?"),  # The NEW question
    ]
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.return_value = "The capital is Paris."

        # Pass the history that INCLUDES the new question
        build_assistant_reply(
            settings,
            "What is its capital?",
            fake_chunks,
            fake_index,
            fake_model,
            history_with_new_question,
        )

    # The history passed to answer_question must NOT include the current question
    mock_answer.assert_called_once()
    call_args = mock_answer.call_args
    passed_history = call_args[1]["history"]

    # The passed history should only have the PRIOR messages, not the new question
    assert len(passed_history) == 2
    assert passed_history[-1].content == "France is a country in Europe."


def test_duplicate_questions_preserve_history(settings, fake_chunks, fake_index, fake_model):
    """When a user asks the same question twice, both must remain in history.

    A content-based filter would strip both occurrences, leaving an orphaned
    assistant reply with no preceding question. The position-only safety net
    only checks the last entry, so earlier identical questions are preserved.
    """
    # History has the same question asked twice
    history = [
        ChatMessage(role="user", content="What is the capital?"),
        ChatMessage(role="assistant", content="Paris is the capital of France."),
        ChatMessage(role="user", content="What is the capital?"),  # Asked again
    ]
    retrieved = [RetrievedChunk(chunk=fake_chunks[0], score=0.9)]

    with (
        patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve,
        patch("pdf_assistant.ui.chat.has_sufficient_context") as mock_has_sufficient,
        patch("pdf_assistant.ui.chat.answer_question") as mock_answer,
    ):
        mock_retrieve.return_value = retrieved
        mock_has_sufficient.return_value = True
        mock_answer.return_value = "Paris."

        # Ask the same question a third time
        build_assistant_reply(
            settings,
            "What is the capital?",
            fake_chunks,
            fake_index,
            fake_model,
            history,
        )

    # The history passed to the model should have 2 messages: [Q, A].
    # The safety net strips the last Q (which matches the current question),
    # but leaves the earlier Q intact.
    mock_answer.assert_called_once()
    call_args = mock_answer.call_args
    passed_history = call_args[1]["history"]

    assert len(passed_history) == 2
    assert passed_history[0].role == "user"
    assert passed_history[0].content == "What is the capital?"
    assert passed_history[1].role == "assistant"
    assert passed_history[1].content == "Paris is the capital of France."


def test_retrieval_error_is_also_caught(settings, fake_chunks, fake_index, fake_model):
    """RetrievalError must be caught and turned into a readable message."""
    with patch("pdf_assistant.ui.chat.retrieve") as mock_retrieve:
        mock_retrieve.side_effect = RetrievalError(
            "The vector index holds 5 vectors but 3 chunks were supplied."
        )

        reply = build_assistant_reply(
            settings, "Where is Paris?", fake_chunks, fake_index, fake_model, []
        )

    assert reply.role == "assistant"
    assert "vector index" in reply.content.lower()
    assert reply.sources == []
