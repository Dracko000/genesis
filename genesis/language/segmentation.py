"""Very small deterministic text segmentation.

This is deliberately not a linguistic parser. It only separates a string into
whitespace-delimited word-like chunks and individual punctuation marks, which
is enough for a transparent baseline. No semantic information is injected.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def segment_text(text: str) -> list[str]:
    """Split text into word-like chunks and punctuation.

    >>> segment_text("A cat, sitting.")
    ['A', 'cat', ',', 'sitting', '.']
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _WORD_RE.findall(text)
