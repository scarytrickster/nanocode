"""Centralized size limiting for tool output.

A single tool call can return an unbounded amount of text: `bash` running
`find /`, a grep with tens of thousands of matches, a read of a generated
file. Any one of those can overflow the model's context window on its own,
which directory filtering (see `tools/ignore.py`) cannot prevent.

Every tool that can return large text funnels its result through
`limit_tool_output`, so the bound lives in one place.

Truncation is never silent: the returned text carries an explicit notice with
the original and returned character counts, so the agent knows it did not see
the whole thing and can narrow its next command instead of drawing conclusions
from a partial result.
"""

from __future__ import annotations

from dataclasses import dataclass

# Conservative default, in characters (this phase does not count tokens).
# Roughly 5k tokens of English text, which leaves plenty of room in a
# 262,144-token context even for a long tool-heavy conversation.
DEFAULT_MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class LimitedOutput:
    """The bounded text plus what had to be dropped to bound it.

    `returned_chars` counts the payload characters kept, not the truncation
    notice appended to them.
    """

    text: str
    truncated: bool
    original_chars: int
    returned_chars: int


def _truncation_notice(original_chars: int, returned_chars: int) -> str:
    """The marker appended to truncated output. Deterministic by design."""

    return (
        "\n\n[OUTPUT TRUNCATED]\n"
        f"Original: {original_chars:,} characters\n"
        f"Returned: {returned_chars:,} characters\n"
        "The output above is the beginning only. "
        "Narrow the command, pattern, or file range to see the rest."
    )


def limit_output(
    text: str | None,
    max_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> LimitedOutput:
    """Bound `text` to `max_chars`, reporting whether anything was dropped.

    The beginning of the output is preserved: it is where headers, error
    messages, and the first matches live.
    """

    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = "" if text is None else text
    original_chars = len(text)

    if original_chars <= max_chars:
        return LimitedOutput(
            text=text,
            truncated=False,
            original_chars=original_chars,
            returned_chars=original_chars,
        )

    kept = text[:max_chars]

    return LimitedOutput(
        text=kept + _truncation_notice(original_chars, max_chars),
        truncated=True,
        original_chars=original_chars,
        returned_chars=max_chars,
    )


def limit_tool_output(
    text: str | None,
    max_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Bound tool output for the string-returning `Tool.execute` interface."""

    return limit_output(text, max_chars).text
