"""Document summarization and explanation service.

Summaries and guided explanations for the Overview and Understand Document
sections. Both use a map-reduce pattern to handle documents too long for the
model's context window:

1. **Map step**: Split chunks into batches where each batch's combined text
   stays under MAX_BATCH_CHARS. Process each batch independently (summarize it,
   or explain it according to the instruction).

2. **Reduce step**: Combine the batch results into one final answer. For a guided
   explanation the instruction is restated at this step, because the user's request
   must govern the combined answer and not just the individual batches.

WHY MAP-REDUCE MATTERS: A 300-page PDF might be 500,000 characters. Sending
that in one request would blow past the model's context limit and waste credits.
By processing in batches of ~6000 chars, we can handle arbitrarily large
documents while keeping each LLM call small and fast. The reduction step
synthesizes the batch results into a coherent whole.

When there is only one batch (a small document), the reduce step is skipped—
that saves one API call.
"""

from __future__ import annotations

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import TextChunk
from pdf_assistant.prompts.explanation_prompt import (
    build_explanation_messages,
    build_final_explanation_messages,
)
from pdf_assistant.prompts.summary_prompt import (
    build_batch_summary_messages,
    build_final_summary_messages,
)

# Import complete into this module's namespace so tests can monkeypatch it as
# summary_service.complete. This is the standard pytest pattern for injecting
# fakes into functions that call other modules.
from pdf_assistant.services.llm_service import complete

# EXCEPTIONS RAISED BY THIS MODULE (one place for Phase 4's UI to look):
#   - SummaryError - this module's own input validation (no chunks to work with).
#   - LlmError     - re-raised from llm_service when any API call fails.
# Both subclass RuntimeError, so the UI can catch (SummaryError, LlmError) - or
# just RuntimeError - and be sure nothing escapes as a raw Streamlit traceback.


class SummaryError(RuntimeError):
    """Raised when a summary or explanation cannot be attempted at all.

    This subclasses ``RuntimeError`` rather than ``ValueError`` so the Phase 4
    UI, which wraps the Overview summary and the eight explanation buttons in
    ``except (SummaryError, LlmError)``, cannot miss it. A ``ValueError`` is not
    a ``RuntimeError`` and would escape as an unhandled Streamlit traceback.
    """


# Maximum combined character count for chunks in one batch. This is a
# conservative limit: Groq's models handle 32k+ tokens, but we stay well below
# that to leave room for the prompt template, instruction text, and the
# response. A batch of 6000 characters is roughly 1500-2000 tokens depending on
# the text, which keeps each request fast and leaves headroom.
MAX_BATCH_CHARS: int = 6000

# Token budget for every call this module makes, larger than llm_service's 800
# default. That default was chosen for short, focused Q&A answers; a whole-document
# summary or guided explanation is much longer prose, and the reduce step over many
# batch summaries is the longest-output path in the project. The default model is a
# reasoning model that spends part of its budget on hidden reasoning before emitting
# anything, and llm_service.complete treats an exhausted budget (empty content) as a
# user-facing error - so an under-sized budget here shows up as a failure, not as a
# merely short answer.
GENERATION_TOKENS: int = 1500


def batch_chunks(
    chunks: list[TextChunk], max_chars: int = MAX_BATCH_CHARS
) -> list[list[TextChunk]]:
    """Group consecutive chunks so each batch stays at or under the character limit.

    This is the first step of map-reduce: decide how to split the document into
    manageable pieces. Chunks are kept in their original order so the narrative
    flow is preserved within each batch.

    SPECIAL CASE: If a single chunk is longer than max_chars, it becomes its own
    batch rather than being silently dropped. Discarding content would produce
    incomplete summaries, which is worse than sending one oversized batch.

    Args:
        chunks: All chunks from the document, in document order.
        max_chars: Maximum combined ``char_count`` for chunks in one batch.
            Defaults to :const:`MAX_BATCH_CHARS`.

    Returns:
        A list of batches (each batch is a list of chunks). The order of chunks
        is preserved: flattening the result gives back the original ``chunks``
        list.
    """
    if not chunks:
        return []

    batches = []
    current_batch = []
    current_batch_chars = 0

    for chunk in chunks:
        chunk_chars = chunk.char_count

        # If adding this chunk would exceed the limit AND we already have chunks
        # in the current batch, finalize the current batch and start a new one.
        if current_batch and current_batch_chars + chunk_chars > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_batch_chars = 0

        # Add the chunk to the current batch. This handles the oversized-chunk
        # case: if the chunk is longer than max_chars, it becomes its own batch
        # because current_batch is empty at this point.
        current_batch.append(chunk)
        current_batch_chars += chunk_chars

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


def summarise_document(settings: Settings, chunks: list[TextChunk]) -> str:
    """Generate a summary of the entire document.

    Uses map-reduce: the document is split into batches (see :func:`batch_chunks`),
    each batch is summarized independently, and the batch summaries are combined
    into one final summary. When there is only one batch, the reduce step is
    skipped and the single batch summary is returned directly.

    Args:
        settings: Configuration including the API key and model name.
        chunks: All chunks from the document, in document order.

    Returns:
        A plain-English summary of the document, written for a BCA student.

    Raises:
        SummaryError: When ``chunks`` is empty. There must be at least one chunk
            to summarize.
        LlmError: When any API call fails. This propagates from
            :func:`llm_service.complete`.
    """
    if not chunks:
        raise SummaryError("Cannot summarize: chunks must contain at least one chunk.")

    # Step 1: Batch the chunks
    batches = batch_chunks(chunks)

    # Step 2 (map): Summarize each batch
    batch_summaries = []
    for batch in batches:
        # Concatenate the text from all chunks in this batch
        batch_text = "\n\n".join(chunk.text for chunk in batch)
        # Build the prompt and call the model
        messages = build_batch_summary_messages(batch_text)
        summary = complete(settings, messages, max_completion_tokens=GENERATION_TOKENS)
        batch_summaries.append(summary)

    # Step 3 (reduce): Combine batch summaries into one final summary.
    # OPTIMIZATION: If there's only one batch, skip the combine step and return
    # its summary directly. That saves one API call, which matters on the free tier.
    if len(batch_summaries) == 1:
        return batch_summaries[0]

    # Multiple batches: combine them
    messages = build_final_summary_messages(batch_summaries)
    return complete(settings, messages, max_completion_tokens=GENERATION_TOKENS)


def explain_document(settings: Settings, chunks: list[TextChunk], instruction: str) -> str:
    """Generate a guided explanation of the document.

    This powers the eight guided explanations in the Understand Document section
    (Key Concepts, Simplify, etc.). It uses the same map-reduce pattern as
    :func:`summarise_document`, but the instruction is passed through to the
    prompt builder at each step.

    Args:
        settings: Configuration including the API key and model name.
        chunks: All chunks from the document, in document order.
        instruction: How to transform the text. Examples:
            - "Explain the key concepts in simple terms"
            - "Summarize the main argument in one paragraph"

    Returns:
        The model's answer, following the instruction.

    Raises:
        SummaryError: When ``chunks`` is empty. There must be at least one chunk.
        LlmError: When any API call fails.
    """
    if not chunks:
        raise SummaryError("Cannot explain: chunks must contain at least one chunk.")

    # Step 1: Batch the chunks
    batches = batch_chunks(chunks)

    # Step 2 (map): Explain each batch according to the instruction
    batch_explanations = []
    for batch in batches:
        # Concatenate the text from all chunks in this batch
        batch_text = "\n\n".join(chunk.text for chunk in batch)
        # Build the prompt with the instruction and call the model
        messages = build_explanation_messages(instruction, batch_text)
        explanation = complete(settings, messages, max_completion_tokens=GENERATION_TOKENS)
        batch_explanations.append(explanation)

    # Step 3 (reduce): Combine batch explanations.
    # Same optimization as summarise_document: skip the combine step if there's
    # only one batch.
    if len(batch_explanations) == 1:
        return batch_explanations[0]

    # Multiple batches: combine them with a reduce prompt that RESTATES the
    # instruction. Reusing build_final_summary_messages here would be a bug: its
    # system message asks for "a single, coherent overview of the entire
    # document", which throws the instruction away and turns every guided
    # explanation into a generic summary. MAX_BATCH_CHARS is only three or four
    # pages, so this is the common path for real PDFs, not an edge case.
    messages = build_final_explanation_messages(instruction, batch_explanations)
    return complete(settings, messages, max_completion_tokens=GENERATION_TOKENS)
