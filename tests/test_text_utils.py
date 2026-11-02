"""Tests for the text helper functions.

Characters that look like plain ASCII are written as escape codes throughout.
A literal non-breaking space is indistinguishable from a normal space on
screen, so a test written with a literal could be silently turned into
``clean_text("a b") == "a b"`` by any tool that normalises whitespace - and it
would still pass while testing nothing.
"""

from __future__ import annotations

import pytest

from pdf_assistant.utils.text_utils import build_excerpt, clean_text, format_file_size


def test_clean_text_collapses_repeated_spaces():
    assert clean_text("hello     world") == "hello world"


def test_clean_text_keeps_paragraph_breaks():
    """Blank lines are a useful hint about where to split chunks."""
    cleaned = clean_text("First paragraph.\n\nSecond paragraph.")

    assert cleaned == "First paragraph.\n\nSecond paragraph."


def test_clean_text_collapses_large_vertical_gaps():
    cleaned = clean_text("Top.\n\n\n\n\nBottom.")

    assert cleaned == "Top.\n\nBottom."


def test_clean_text_rejoins_hyphenated_line_breaks():
    """PDF extraction often splits words across lines with a hyphen."""
    assert clean_text("informa-\ntion") == "information"


def test_clean_text_normalises_non_breaking_space():
    assert clean_text("a\xa0b") == "a b"


def test_clean_text_removes_zero_width_characters():
    assert clean_text("wo​rd") == "word"
    assert clean_text("﻿text") == "text"


def test_clean_text_normalises_curly_quotes():
    assert clean_text("“quoted”") == '"quoted"'
    assert clean_text("it’s") == "it's"


def test_clean_text_normalises_dashes_and_ellipsis():
    assert clean_text("a – b") == "a - b"
    assert clean_text("a — b") == "a - b"
    assert clean_text("wait…") == "wait..."


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


def test_clean_text_strips_surrounding_whitespace():
    assert clean_text("   padded   ") == "padded"


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024 * 1024, "1.0 MB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (2 * 1024 * 1024 * 1024, "2.0 GB"),
    ],
)
def test_format_file_size(size_bytes, expected):
    assert format_file_size(size_bytes) == expected


def test_format_file_size_rejects_negative():
    with pytest.raises(ValueError):
        format_file_size(-1)


def test_build_excerpt_returns_short_text_unchanged():
    assert build_excerpt("Short text.", max_chars=100) == "Short text."


def test_build_excerpt_truncates_long_text():
    excerpt = build_excerpt("word " * 200, max_chars=50)

    assert excerpt.endswith("...")
    # Bounded on both sides: an implementation returning just "..." would pass
    # an upper-bound-only check.
    assert 25 < len(excerpt) <= 53  # up to 50 characters plus the "..." marker


def test_build_excerpt_does_not_split_a_word_in_half():
    text = "alpha beta gamma delta epsilon zeta eta theta"

    excerpt = build_excerpt(text, max_chars=20)

    words = excerpt.removesuffix("...").split()
    # Without this, an empty result would make the all(...) below vacuously true.
    assert words, "excerpt should still contain some words"
    assert all(word in text.split() for word in words)


def test_build_excerpt_collapses_newlines_for_display():
    assert build_excerpt("line one\nline two", max_chars=100) == "line one line two"


def test_build_excerpt_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        build_excerpt("some text", max_chars=0)
