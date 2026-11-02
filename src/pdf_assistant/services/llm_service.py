"""Groq language model service for question answering.

This module wraps all calls to the Groq API and maps every failure mode to
LlmError with a plain-English message safe to show in the UI. The actual API key
is never included in any error message, log, or user-visible text.

CRITICAL BEHAVIOUR FOR REASONING MODELS: The default model (openai/gpt-oss-120b)
is a reasoning model that spends completion tokens on hidden reasoning before
producing the visible answer. If the token budget runs out during reasoning, the
call completes successfully (finish_reason="stop") but message.content is empty.
This module treats empty or whitespace-only content as a failure and raises
LlmError telling the user to retry or increase the token budget.

The reasoning is private and never returned: we only read message.content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    NotFoundError,
    RateLimitError,
)

from pdf_assistant.config import Settings
from pdf_assistant.prompts.qa_prompt import build_qa_messages

if TYPE_CHECKING:
    from pdf_assistant.models.schemas import ChatMessage, RetrievedChunk


class LlmError(RuntimeError):
    """Raised when the language model call fails for any reason.

    The message is always safe to show in the UI: it never contains the API key,
    never exposes internal state, and always suggests what the user should do.
    """


def complete(
    settings: Settings,
    messages: list[dict[str, str]],
    max_completion_tokens: int = 800,
) -> str:
    """Call the Groq API and return the model's text answer.

    Args:
        settings: Configuration including the API key and model name. The key
            must be present (``settings.has_api_key == True``), or this raises
            before attempting any network call.
        messages: The conversation formatted as a list of message dicts with
            "role" and "content" keys, ready for the API. Use
            :func:`build_qa_messages` to construct this for question answering.
        max_completion_tokens: Token budget for the response. The default of 800
            is deliberately generous because reasoning models spend tokens on
            hidden reasoning before emitting the visible answer. A budget that is
            too low produces an empty string with finish_reason="stop", which we
            treat as a failure.

    Returns:
        The model's text answer, stripped of leading and trailing whitespace.

    Raises:
        LlmError: When the API key is missing or blank, when any API call fails,
            or when the model returns empty or whitespace-only content (the latter
            happens with reasoning models when the token budget is exhausted).
    """
    # Check for a missing API key before constructing the client, so the key
    # never reaches the library and cannot appear in any traceback or log.
    if not settings.has_api_key:
        raise LlmError(
            "Groq API key is not configured. Please add GROQ_API_KEY to your .env file. "
            "You can obtain a free API key from https://console.groq.com/."
        )

    # Construct the client lazily so that an absent key never reaches the Groq
    # library and Settings objects without keys can still be created for testing.
    try:
        client = Groq(api_key=settings.groq_api_key)
    except Exception as exc:
        # Should not happen given the has_api_key check, but defensive.
        raise LlmError(
            "Failed to initialise Groq client. Check your GROQ_API_KEY is valid."
        ) from exc

    # Call the API, mapping every known failure mode to LlmError.
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
    except AuthenticationError as exc:
        # Key is present but rejected. Most likely it is expired, revoked, or
        # mistyped. DO NOT include the key or any part of the original message
        # that might contain it.
        raise LlmError(
            "Authentication failed. Your GROQ_API_KEY was rejected by the server. "
            "Check that it is correct and active at https://console.groq.com/keys."
        ) from exc
    except RateLimitError as exc:
        # Free-tier limit reached. This is temporary and the user can retry.
        raise LlmError(
            "Rate limit reached. Groq's free tier has a request limit. "
            "Please wait a few minutes and try again, or upgrade your plan at "
            "https://console.groq.com/settings/billing."
        ) from exc
    except NotFoundError as exc:
        # Model not found. The configured model name is wrong or the model was
        # retired. Mention GROQ_MODEL so the user knows which setting to fix.
        raise LlmError(
            f"Model not found. GROQ_MODEL is set to '{settings.groq_model}', but that "
            f"model is not available. Check the current list of models at "
            f"https://console.groq.com/docs/models and update your .env file."
        ) from exc
    except APITimeoutError as exc:
        # Network timeout. Could be the user's connection or Groq's servers.
        raise LlmError(
            "Request timed out. Check your internet connection and try again. "
            "If the problem persists, Groq's servers may be experiencing issues."
        ) from exc
    except APIConnectionError as exc:
        # Network connection failed. Similar to timeout but more clearly a connection issue.
        raise LlmError(
            "Could not connect to Groq. Check your internet connection and try again."
        ) from exc
    except APIError as exc:
        # Generic API error. This catches anything not covered above (e.g.
        # InternalServerError, BadRequestError, UnprocessableEntityError).
        # Include the status code if available, but never the full message
        # because it might contain the key if the error came from authentication.
        status = getattr(exc, "status_code", None)
        status_text = f" (status {status})" if status else ""
        raise LlmError(
            f"Groq API error{status_text}. This might be a temporary problem with "
            f"Groq's servers. Please try again in a few moments."
        ) from exc

    # Extract the answer. We only read message.content; message.reasoning is the
    # model's private thinking and must never be shown or returned.
    try:
        content = response.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        raise LlmError(
            "Received an invalid response from Groq. The response structure was "
            "not as expected. Please try again."
        ) from exc

    # CRITICAL: reasoning models can return empty content when the token budget
    # runs out. finish_reason is still "stop", so the call looks successful, but
    # there is no answer. Treat this as a failure rather than silently returning
    # an empty string that would appear as a blank message in the UI.
    if not content or not content.strip():
        raise LlmError(
            "The model returned an empty response. This can happen with reasoning models "
            "when the token budget is exhausted. Try asking a simpler question, or ask "
            "an administrator to increase the max_completion_tokens setting."
        )

    return content.strip()


def answer_question(
    settings: Settings,
    question: str,
    retrieved: list[RetrievedChunk],
    history: list[ChatMessage],
) -> str:
    """Answer a question by combining retrieval with language model generation.

    This is a convenience function that composes :func:`build_qa_messages` from
    Task 4 with :func:`complete`. It builds the grounded prompt with document
    excerpts and conversation history, calls the model, and returns the answer.

    Args:
        settings: Configuration including the API key and model name.
        question: The user's question.
        retrieved: Chunks retrieved by the vector search. These are formatted
            into the prompt as reference material. Their page numbers are NOT
            included in the prompt — the UI builds citations separately.
        history: Previous conversation turns. Only the last 4 question-and-answer
            exchanges are included in the prompt.

    Returns:
        The model's text answer, which might be a genuine answer or the exact
        abstention message when the retrieved context is insufficient.

    Raises:
        LlmError: When the API call fails for any reason.
    """
    messages = build_qa_messages(question, retrieved, history)
    return complete(settings, messages)
