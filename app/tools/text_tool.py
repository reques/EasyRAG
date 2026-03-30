"""Text processing tool – word count, char count, summarise, clean."""
from __future__ import annotations

import re
from typing import Any, Dict

from app.core.exceptions import ToolExecutionError
from app.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_OPERATIONS = {
    "word_count",
    "char_count",
    "sentence_count",
    "clean",
    "uppercase",
    "lowercase",
    "reverse",
    "extract_numbers",
    "stats",
}


def text_tool(operation: str, text: str, **kwargs: Any) -> str:
    """Perform a text processing *operation* on *text*.

    Args:
        operation: One of the SUPPORTED_OPERATIONS.
        text:      The input text.
        **kwargs:  Additional parameters (currently unused).

    Returns:
        A human-readable string with the result.

    Raises:
        ToolExecutionError: for unknown operations or empty text.
    """
    logger.debug("text_tool op=%s text_len=%d", operation, len(text))
    if not text.strip():
        raise ToolExecutionError("Input text is empty.")
    op = operation.lower().strip()
    if op not in SUPPORTED_OPERATIONS:
        raise ToolExecutionError(
            f"Unknown operation '{op}'. Supported: {sorted(SUPPORTED_OPERATIONS)}"
        )
    if op == "word_count":
        return f"Word count: {len(text.split())}"
    if op == "char_count":
        return f"Character count (with spaces): {len(text)}  |  without spaces: {len(text.replace(' ', ''))}"
    if op == "sentence_count":
        sentences = re.split(r'[.!?。！？]+', text)
        count = len([s for s in sentences if s.strip()])
        return f"Sentence count: {count}"
    if op == "clean":
        cleaned = re.sub(r'\s+', ' ', text).strip()
        return f"Cleaned text:\n{cleaned}"
    if op == "uppercase":
        return text.upper()
    if op == "lowercase":
        return text.lower()
    if op == "reverse":
        return text[::-1]
    if op == "extract_numbers":
        nums = re.findall(r'-?\d+\.?\d*', text)
        return f"Numbers found: {', '.join(nums) if nums else 'none'}"
    if op == "stats":
        words = text.split()
        sentences = re.split(r'[.!?。！？]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        result = (
            f"Text statistics:\n"
            f"  Characters : {len(text)}\n"
            f"  Words      : {len(words)}\n"
            f"  Sentences  : {sentence_count}\n"
            f"  Avg word length: {avg_word_len:.1f} chars"
        )
        return result
    raise ToolExecutionError(f"Unhandled operation: {op}")
