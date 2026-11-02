"""Small text helpers used by the services and the user interface."""

from __future__ import annotations

import re

# Matches runs of spaces and tabs, but deliberately NOT newlines, because
# newlines are handled separately to keep paragraph structure.
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\r\f\v]+")

# Matches three or more newlines, used to collapse large vertical gaps.
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Characters PDF files commonly use that confuse text handling later.
#
# Every key is written as an escape code on purpose. Three of them are
# invisible, so a literal would make this table impossible to read and easy to
# break by accident; the rest follow the same style for consistency.
_REPLACEMENTS = {
    " ": " ",  # non-breaking space  -> normal space
    "​": "",  # zero-width space    -> removed
    "﻿": "",  # byte-order mark     -> removed
    "‘": "'",  # left single quote   -> straight apostrophe
    "’": "'",  # right single quote  -> straight apostrophe
    "“": '"',  # left double quote   -> straight quote
    "”": '"',  # right double quote  -> straight quote
    "–": "-",  # en dash             -> hyphen
    "—": "-",  # em dash             -> hyphen
    "…": "...",  # ellipsis          -> three dots
}


def clean_text(text: str) -> str:
    """Tidy up text extracted from a PDF.

    PDF extraction produces messy output: hyphenated words broken across lines,
    non-breaking spaces, and large blank gaps where images or headers sat. This
    cleanup matters for retrieval quality, because the embedding model compares
    the question against this text. Noise makes the comparison worse.

    Paragraph breaks (blank lines) are kept, because they are a useful hint for
    the chunker about where a sensible split point is.

    Args:
        text: Raw text straight out of PyMuPDF.

    Returns:
        Cleaned text, with leading and trailing whitespace removed.
    """
    if not text:
        return ""

    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Re-join words split across a line break by hyphenation, e.g.
    # "informa-\ntion" becomes "information".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Squeeze horizontal whitespace, then trim each line.
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse big vertical gaps down to a single blank line.
    text = _EXCESS_NEWLINES.sub("\n\n", text)

    return text.strip()


def format_file_size(size_bytes: int) -> str:
    """Turn a byte count into something readable, e.g. ``"1.4 MB"``."""
    if size_bytes < 0:
        raise ValueError("size_bytes cannot be negative")
    if size_bytes < 1024:
        return f"{size_bytes} B"

    size = float(size_bytes)
    for unit in ("KB", "MB", "GB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TB"


def build_excerpt(text: str, max_chars: int = 300) -> str:
    """Shorten a chunk for display as a source excerpt under an answer.

    Cuts at the last space before the limit so a word is not sliced in half.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed

    cut = collapsed[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        cut = cut[:last_space]
    return cut.rstrip() + "..."
