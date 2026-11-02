"""Tests for document summary and explanation generation.

These tests verify that summaries and explanations use the map-reduce pattern
correctly: large documents are batched, each batch is processed, and batch
results are combined. Tests inject a fake `complete` function to verify the
right prompts are sent without making real API calls.
"""

from __future__ import annotations

import pytest

from pdf_assistant.config import Settings
from pdf_assistant.models.schemas import TextChunk
from pdf_assistant.services import summary_service
from pdf_assistant.services.summary_service import (
    SummaryError,
    batch_chunks,
    explain_document,
    summarise_document,
)


@pytest.fixture
def fake_settings() -> Settings:
    """A Settings object with all required fields."""
    return Settings(
        groq_api_key="fake-key",
        groq_model="fake-model",
        embedding_model="fake-embedding",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        max_upload_size_mb=20,
        min_similarity=0.25,
    )


def test_batch_chunks_keeps_batches_within_the_budget():
    """Each batch's combined text stays at or under the character limit."""
    chunks = [
        TextChunk(chunk_id=0, text="a" * 2000, page_number=1),
        TextChunk(chunk_id=1, text="b" * 2000, page_number=1),
        TextChunk(chunk_id=2, text="c" * 2000, page_number=2),
        TextChunk(chunk_id=3, text="d" * 2000, page_number=2),
    ]

    batches = batch_chunks(chunks, max_chars=4500)

    # Should split into 2 batches: [0,1] and [2,3]
    assert len(batches) == 2
    # Check each batch respects the limit
    for batch in batches:
        total_chars = sum(chunk.char_count for chunk in batch)
        assert total_chars <= 4500


def test_batch_chunks_puts_everything_in_one_batch_when_it_fits():
    """When the document is small enough, all chunks go in one batch."""
    chunks = [
        TextChunk(chunk_id=0, text="a" * 500, page_number=1),
        TextChunk(chunk_id=1, text="b" * 500, page_number=1),
    ]

    batches = batch_chunks(chunks, max_chars=2000)

    assert len(batches) == 1
    assert batches[0] == chunks


def test_batch_chunks_splits_a_long_document_into_several():
    """A document larger than the budget is split into multiple batches."""
    chunks = [TextChunk(chunk_id=i, text="x" * 1000, page_number=1) for i in range(10)]

    batches = batch_chunks(chunks, max_chars=3000)

    # Each chunk is 1000 chars, so with a 3000 char limit, we get 4 batches
    assert len(batches) >= 4
    # Verify all chunks are present
    all_chunks = [chunk for batch in batches for chunk in batch]
    assert all_chunks == chunks


def test_batch_chunks_keeps_an_oversized_single_chunk():
    """A chunk longer than the budget becomes its own batch rather than being dropped."""
    oversized = TextChunk(chunk_id=0, text="z" * 7000, page_number=1)
    normal = TextChunk(chunk_id=1, text="a" * 500, page_number=2)
    chunks = [oversized, normal]

    batches = batch_chunks(chunks, max_chars=6000)

    # The oversized chunk should be in its own batch
    assert len(batches) == 2
    assert batches[0] == [oversized]
    assert batches[1] == [normal]


def test_batch_chunks_preserves_document_order():
    """Chunks appear in batches in the same order they were input."""
    chunks = [
        TextChunk(chunk_id=0, text="first", page_number=1),
        TextChunk(chunk_id=1, text="second", page_number=1),
        TextChunk(chunk_id=2, text="third", page_number=2),
    ]

    batches = batch_chunks(chunks, max_chars=5000)

    # Flatten and compare
    flattened = [chunk for batch in batches for chunk in batch]
    assert flattened == chunks


def test_batch_chunks_of_an_empty_list_is_empty():
    """Batching an empty list returns an empty list."""
    batches = batch_chunks([], max_chars=1000)
    assert batches == []


def test_summarise_document_makes_one_call_for_a_small_document(fake_settings, monkeypatch):
    """When there's only one batch, skip the reduce step and return directly."""
    chunks = [TextChunk(chunk_id=0, text="small document", page_number=1)]

    call_count = 0

    def fake_complete(settings, messages, max_completion_tokens=800):
        nonlocal call_count
        call_count += 1
        return "Summary of the document"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    result = summarise_document(fake_settings, chunks)

    # Should make exactly 1 call (no reduce step)
    assert call_count == 1
    assert result == "Summary of the document"


def test_summarise_document_combines_batches_for_a_large_document(fake_settings, monkeypatch):
    """For multiple batches, calls equal batches + 1 (map + reduce)."""
    # Create chunks that will force multiple batches
    # Each chunk is 3500 chars, so with 6000 limit: [chunk0], [chunk1], [chunk2]
    chunks = [TextChunk(chunk_id=i, text="x" * 3500, page_number=1) for i in range(3)]

    call_count = 0

    def fake_complete(settings, messages, max_completion_tokens=800):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            # First 3 calls are batch summaries
            return f"Batch summary {call_count}"
        else:
            # 4th call is the final combine
            return "Final combined summary"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    result = summarise_document(fake_settings, chunks)

    # Should make 4 calls: 3 batches + 1 combine
    assert call_count == 4
    assert result == "Final combined summary"


def test_summarise_document_sends_every_chunk_text_somewhere(fake_settings, monkeypatch):
    """No chunk is silently skipped — all text reaches the LLM."""
    chunks = [
        TextChunk(chunk_id=0, text="first chunk", page_number=1),
        TextChunk(chunk_id=1, text="second chunk", page_number=1),
        TextChunk(chunk_id=2, text="third chunk", page_number=2),
    ]

    all_messages = []

    def fake_complete(settings, messages, max_completion_tokens=800):
        all_messages.extend(messages)
        return "Summary"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    summarise_document(fake_settings, chunks)

    # Concatenate all message content
    all_content = " ".join(msg["content"] for msg in all_messages if msg["role"] == "user")

    # Verify every chunk's text appears somewhere
    for chunk in chunks:
        assert chunk.text in all_content


def test_summarise_document_with_no_chunks_raises(fake_settings):
    """Attempting to summarize zero chunks raises SummaryError."""
    with pytest.raises(SummaryError, match="at least one chunk"):
        summarise_document(fake_settings, [])


def test_explain_document_with_no_chunks_raises(fake_settings):
    """Attempting to explain zero chunks raises SummaryError."""
    with pytest.raises(SummaryError, match="at least one chunk"):
        explain_document(fake_settings, [], "Explain this")


def test_summary_error_is_a_runtime_error():
    """Phase 4 wraps the summary and explanation buttons in except (SummaryError, LlmError).

    SummaryError must subclass RuntimeError like LlmError does. A ValueError -
    which is what this guard used to raise - is not a RuntimeError and would
    escape such a clause as a raw Streamlit traceback.
    """
    from pdf_assistant.services.llm_service import LlmError

    assert issubclass(SummaryError, RuntimeError)
    assert issubclass(LlmError, RuntimeError)


def test_explain_document_passes_the_instruction_through(fake_settings, monkeypatch):
    """The instruction is included in the messages sent to the LLM."""
    chunks = [TextChunk(chunk_id=0, text="some text", page_number=1)]

    received_messages = []

    def fake_complete(settings, messages, max_completion_tokens=800):
        received_messages.extend(messages)
        return "Explanation"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    result = explain_document(fake_settings, chunks, "Explain this in simple terms")

    # The instruction should appear in the user message
    all_content = " ".join(msg["content"] for msg in received_messages if msg["role"] == "user")
    assert "Explain this in simple terms" in all_content
    assert result == "Explanation"


def test_explain_document_uses_batching_for_a_large_document(fake_settings, monkeypatch):
    """Large documents are batched in explain_document too."""
    # Create chunks that force multiple batches
    # Each chunk is 3500 chars, so with 6000 limit: [chunk0], [chunk1], [chunk2]
    chunks = [TextChunk(chunk_id=i, text="y" * 3500, page_number=1) for i in range(3)]

    call_count = 0

    def fake_complete(settings, messages, max_completion_tokens=800):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return f"Explanation batch {call_count}"
        else:
            return "Final explanation"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    result = explain_document(fake_settings, chunks, "Simplify this")

    # Should use batching: 3 batches + 1 combine = 4 calls
    assert call_count == 4
    assert result == "Final explanation"


def test_explain_document_restates_the_instruction_in_the_reduce_step(fake_settings, monkeypatch):
    """The reduce step must still be governed by the user's instruction.

    The reduce step used to borrow build_final_summary_messages, whose system
    message asks for "a single, coherent overview of the entire document". That
    silently discarded the instruction, so all eight guided explanations
    collapsed into a generic summary. Because MAX_BATCH_CHARS is only three or
    four pages, that hit most real PDFs. Counting calls (the test above) cannot
    see this, so this test inspects the final call's messages.
    """
    # Each chunk is 3500 chars, so with the 6000 limit this forces three batches
    # plus a reduce step.
    chunks = [TextChunk(chunk_id=i, text="y" * 3500, page_number=1) for i in range(3)]
    instruction = "Explain the difficult terms in this document"

    calls: list[list[dict[str, str]]] = []

    def fake_complete(settings, messages, max_completion_tokens=800):
        calls.append(messages)
        return f"Explanation part {len(calls)}"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    explain_document(fake_settings, chunks, instruction)

    # 3 map calls + 1 reduce call; the reduce call is the last one.
    assert len(calls) == 4
    reduce_messages = calls[-1]
    reduce_content = " ".join(msg["content"] for msg in reduce_messages)

    # The instruction must govern the combined answer.
    assert instruction in reduce_content
    # And the generic summary wording must NOT be what drives it.
    assert "coherent overview of the entire document" not in reduce_content
    # Every batch result must still be carried into the reduce step.
    for part_number in (1, 2, 3):
        assert f"Explanation part {part_number}" in reduce_content


def test_summary_and_explanation_use_the_larger_token_budget(fake_settings, monkeypatch):
    """Summaries are long prose, so they must not take the 800-token Q&A default.

    llm_service.complete treats an exhausted budget as a user-facing error (a
    reasoning model returns empty content), so an under-sized budget on the
    longest-output path in the project surfaces as a failure rather than as a
    merely short answer.
    """
    # Three batches plus a reduce step, so both the map and the reduce call are
    # covered.
    chunks = [TextChunk(chunk_id=i, text="z" * 3500, page_number=1) for i in range(3)]

    budgets: list[int] = []

    def fake_complete(settings, messages, max_completion_tokens=800):
        budgets.append(max_completion_tokens)
        return "text"

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    summarise_document(fake_settings, chunks)
    explain_document(fake_settings, chunks, "Explain this")

    assert summary_service.GENERATION_TOKENS > 800
    assert budgets == [summary_service.GENERATION_TOKENS] * 8


def test_llm_failure_propagates_as_llm_error(fake_settings, monkeypatch):
    """When the LLM call raises LlmError, it propagates to the caller."""
    from pdf_assistant.services.llm_service import LlmError

    chunks = [TextChunk(chunk_id=0, text="some text", page_number=1)]

    def fake_complete(settings, messages, max_completion_tokens=800):
        raise LlmError("Model not found")

    monkeypatch.setattr(summary_service, "complete", fake_complete)

    with pytest.raises(LlmError, match="Model not found"):
        summarise_document(fake_settings, chunks)

    with pytest.raises(LlmError, match="Model not found"):
        explain_document(fake_settings, chunks, "Explain this")
