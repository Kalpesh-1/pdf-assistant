"""Tests for the LLM service that calls Groq.

CRITICAL: No test in this file may contact the network or make a real Groq call.
Every test mocks groq.Groq. A real call would cost the student credits and make
the suite non-deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import ChatMessage, RetrievedChunk, TextChunk
from pdf_assistant.services import llm_service
from pdf_assistant.services.llm_service import LlmError


def _make_settings(api_key: str = "test-key") -> Settings:
    """Build a minimal Settings object for testing."""
    return Settings(
        groq_api_key=api_key,
        groq_model="test-model",
        embedding_model="test-embedding",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )


def _make_mock_response(
    content: str, finish_reason: str = "stop", reasoning: str = ""
) -> MagicMock:
    """Build a mock Groq response object."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.reasoning = reasoning

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


def test_complete_returns_the_message_content():
    """The model's text answer is extracted and returned."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("Test answer")

        result = llm_service.complete(settings, messages)

    assert result == "Test answer"


def test_complete_strips_surrounding_whitespace():
    """Leading and trailing whitespace is removed from the model's reply."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "  Answer with spaces  "
        )

        result = llm_service.complete(settings, messages)

    assert result == "Answer with spaces"


def test_complete_passes_the_configured_model():
    """The model name from settings reaches the API call."""
    settings = _make_settings()
    settings = Settings(
        groq_api_key="key",
        groq_model="specific-model-name",
        embedding_model="test-embedding",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("Answer")

        llm_service.complete(settings, messages)

        # Verify the model name was passed to the API
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "specific-model-name"


def test_complete_passes_the_token_budget():
    """The max_completion_tokens parameter reaches the API call."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("Answer")

        llm_service.complete(settings, messages, max_completion_tokens=500)

        # Verify the token budget was passed to the API
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 500


def test_missing_api_key_raises_before_any_call():
    """When the API key is missing, LlmError is raised before constructing the client."""
    settings = _make_settings(api_key="")
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

        # The Groq client was never constructed
        MockGroq.assert_not_called()

    # The error message tells the user how to fix it
    assert "GROQ_API_KEY" in str(exc_info.value)
    assert ".env" in str(exc_info.value)


def test_blank_content_raises_llm_error():
    """Empty content from a reasoning model is treated as a failure."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        # Simulate a reasoning model that exhausted its token budget on reasoning
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "", finish_reason="stop"
        )

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message tells the user what happened
    assert "empty" in str(exc_info.value).lower() or "blank" in str(exc_info.value).lower()


def test_whitespace_only_content_raises_llm_error():
    """Whitespace-only content is also treated as a failure."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        # Simulate a reasoning model that returned only whitespace
        mock_client.chat.completions.create.return_value = _make_mock_response("   \n\t  ")

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message tells the user what happened
    assert "empty" in str(exc_info.value).lower() or "blank" in str(exc_info.value).lower()


def test_reasoning_field_is_never_returned():
    """Only message.content is returned; message.reasoning is ignored."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        # Simulate a reasoning model with separate reasoning and content
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="Answer",
            reasoning="Secret internal thinking that must never be shown",
        )

        result = llm_service.complete(settings, messages)

    # Only the content is returned
    assert result == "Answer"
    assert "Secret" not in result
    assert "thinking" not in result


def test_rate_limit_error_becomes_a_readable_message():
    """RateLimitError is caught and mapped to a user-friendly message."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Import the real exception class
        from groq import RateLimitError

        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit reached",
            response=MagicMock(status_code=429),
            body=None,
        )

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message is helpful
    message = str(exc_info.value).lower()
    assert "rate" in message or "limit" in message


def test_authentication_error_becomes_a_readable_message():
    """AuthenticationError is caught and mapped to a user-friendly message."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Import the real exception class
        from groq import AuthenticationError

        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message is helpful
    message = str(exc_info.value).lower()
    assert "key" in message or "auth" in message


def test_timeout_error_becomes_a_readable_message():
    """APITimeoutError is caught and mapped to a user-friendly message."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Import the real exception class
        from groq import APITimeoutError

        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message is helpful
    message = str(exc_info.value).lower()
    assert "timeout" in message or "network" in message or "connection" in message


def test_model_not_found_mentions_GROQ_MODEL():
    """NotFoundError is caught and mentions the GROQ_MODEL setting."""
    settings = _make_settings()
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Import the real exception class
        from groq import NotFoundError

        mock_client.chat.completions.create.side_effect = NotFoundError(
            "Model not found",
            response=MagicMock(status_code=404),
            body=None,
        )

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The error message mentions the setting and the console URL
    message = str(exc_info.value)
    assert "GROQ_MODEL" in message


def test_error_messages_never_contain_the_api_key():
    """No exception text ever exposes the API key, even partially."""
    secret_key = "gsk_secret_key_that_must_never_appear_1234567890"
    settings = _make_settings(api_key=secret_key)
    messages = [{"role": "user", "content": "Hello"}]

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client

        # Import the real exception class
        from groq import AuthenticationError

        mock_client.chat.completions.create.side_effect = AuthenticationError(
            f"Authentication failed with key {secret_key}",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(LlmError) as exc_info:
            llm_service.complete(settings, messages)

    # The secret key does not appear in the error message
    message = str(exc_info.value)
    assert secret_key not in message
    assert "gsk_" not in message


def test_answer_question_sends_the_retrieved_excerpts():
    """The answer_question convenience function passes retrieved chunks to the prompt."""
    settings = _make_settings()
    question = "What is the answer?"
    chunk = TextChunk(chunk_id=0, text="The answer is 42.", page_number=1)
    retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]
    history: list[ChatMessage] = []

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("42")

        result = llm_service.answer_question(settings, question, retrieved, history)

    # The answer is returned
    assert result == "42"

    # The messages sent to the API contain the chunk text
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages_content = " ".join(msg["content"] for msg in call_kwargs["messages"])
    assert "The answer is 42." in messages_content


def test_answer_question_returns_the_abstention_string_unchanged():
    """When the model legitimately abstains, the exact string is returned."""
    from pdf_assistant.prompts.qa_prompt import INSUFFICIENT_CONTEXT_MESSAGE

    settings = _make_settings()
    question = "What is the answer?"
    retrieved: list[RetrievedChunk] = []
    history: list[ChatMessage] = []

    with patch("pdf_assistant.services.llm_service.Groq") as MockGroq:
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        # Model returns the exact abstention string
        mock_client.chat.completions.create.return_value = _make_mock_response(
            INSUFFICIENT_CONTEXT_MESSAGE
        )

        result = llm_service.answer_question(settings, question, retrieved, history)

    # The abstention string is returned unchanged
    assert result == INSUFFICIENT_CONTEXT_MESSAGE
