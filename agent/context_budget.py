"""Centralized context budgeting for every LLM request.

Phase 6.6 bounded each individual tool result at 20,000 characters. That is
necessary but not sufficient: twenty bounded results still accumulate into
~400,000 characters of conversation, which is how a run reached 371,405 input
tokens against a 262,144-token model limit.

This module is the second layer. It takes the messages a caller is about to
send and returns a version guaranteed to fit a configured token budget, or
raises before any network request happens.

Two properties matter more than compression ratio:

- **Roles are preserved.** A tool result never becomes a system instruction.
  Compression shortens content in place or drops whole exchanges; it never
  re-labels anything, so it cannot turn tool output into an instruction.
- **API shape is preserved.** An assistant message carrying tool_calls and the
  tool messages answering it are treated as one unit and are kept or dropped
  together, so a compressed conversation is still a valid request.

Compression is deterministic and offline. No model is called to summarize
context: that would add cost, another rate-limit surface, and a second context
problem inside the fix for the first one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from config.settings import (
    CHARS_PER_TOKEN,
    CONTEXT_TOKEN_BUDGET,
    MAX_COMPRESSED_ENTRY_CHARS,
    MIN_RECENT_MESSAGES,
)

# Markers that make compression visible to both the model and the reader.
ENTRY_OMISSION_MARKER = "[... {omitted:,} characters omitted ...]"
DIGEST_HEADER = "[CONTEXT COMPRESSED]"

# Share of a compressed entry kept from the start; the rest comes from the end.
# The beginning usually carries the command and the headline, the end usually
# carries the result or the error.
HEAD_SHARE = 0.6


class ContextBudgetError(RuntimeError):
    """Raised when even the required core context cannot fit the budget.

    Raised locally, before any request is made: an oversized prompt never
    reaches the provider.
    """


def estimate_tokens(text: str) -> int:
    """Estimate the tokens in a string. An estimate, not a count.

    No tokenizer is available in this project, so this is a deterministic
    character-based approximation: roughly four characters per token for
    English prose and source code. It is intentionally named `estimate_` --
    nothing here should be read as an exact token count.
    """

    if not text:
        return 0

    return math.ceil(len(text) / CHARS_PER_TOKEN)


def message_chars(message: dict[str, Any]) -> int:
    """Characters a message contributes, including its tool-call payload."""

    total = len(str(message.get("content") or ""))

    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}

        total += len(str(function.get("name", "")))
        total += len(str(function.get("arguments", "")))

    return total


def context_chars(messages: list[dict[str, Any]]) -> int:
    return sum(message_chars(message) for message in messages)


def estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimated tokens for a whole message list. O(n) in total size."""

    return estimate_tokens("x" * context_chars(messages)) if messages else 0


def compress_text(text: str, max_chars: int) -> str:
    """Shorten text while keeping both ends.

    The beginning and the end of a tool result carry the most information --
    the command and the headline at the top, the result or error at the
    bottom. Everything dropped is announced in the middle.
    """

    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    if len(text) <= max_chars:
        return text

    marker = ENTRY_OMISSION_MARKER.format(omitted=len(text) - max_chars)

    budget = max(0, max_chars - len(marker))

    head_chars = int(budget * HEAD_SHARE)
    tail_chars = budget - head_chars

    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""

    return f"{head}\n{marker}\n{tail}"


@dataclass
class ContextUnit:
    """One indivisible piece of conversation.

    An assistant message with tool_calls and the tool messages answering it
    are one unit: dropping half of that pair produces an invalid request.
    """

    messages: list[dict[str, Any]]
    protected: bool = False
    kind: str = "message"

    @property
    def chars(self) -> int:
        return context_chars(self.messages)


@dataclass
class ContextBudgetResult:
    """What budgeting did, in counts and sizes only.

    Never carries prompt text, tool output, or model responses -- this is what
    gets traced.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    original_chars: int = 0
    final_chars: int = 0
    original_estimated_tokens: int = 0
    final_estimated_tokens: int = 0
    compressed: bool = False
    entries_compressed: int = 0
    entries_dropped: int = 0
    compression_passes: int = 0
    hard_capped: bool = False

    def to_dict(self) -> dict:
        return {
            "original_chars": self.original_chars,
            "final_chars": self.final_chars,
            "original_estimated_tokens": self.original_estimated_tokens,
            "final_estimated_tokens": self.final_estimated_tokens,
            "compressed": self.compressed,
            "entries_compressed": self.entries_compressed,
            "entries_dropped": self.entries_dropped,
            "compression_passes": self.compression_passes,
            "hard_capped": self.hard_capped,
        }


class ContextBudgetManager:
    """Keeps the context of one LLM request inside a token budget.

    Holds configuration only -- every `prepare` call works on locals, so
    parallel RLM children can share an instance without sharing state, and
    each child agent gets its own instance anyway.
    """

    def __init__(
        self,
        max_tokens: int = CONTEXT_TOKEN_BUDGET,
        min_recent_messages: int = MIN_RECENT_MESSAGES,
        max_entry_chars: int = MAX_COMPRESSED_ENTRY_CHARS,
        tracer: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        self.max_tokens = max_tokens
        self.min_recent_messages = max(1, min_recent_messages)
        self.max_entry_chars = max(1, max_entry_chars)
        self.tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return messages guaranteed to fit the budget."""

        return self.analyze(messages).messages

    def analyze(self, messages: list[dict[str, Any]]) -> ContextBudgetResult:
        """Budget the context and report what it took."""

        messages = list(messages or [])

        original_chars = context_chars(messages)

        result = ContextBudgetResult(
            messages=messages,
            original_chars=original_chars,
            final_chars=original_chars,
            original_estimated_tokens=estimate_tokens("x" * original_chars),
            final_estimated_tokens=estimate_tokens("x" * original_chars),
        )

        if result.original_estimated_tokens <= self.max_tokens:
            # Under budget: nothing is touched, so a normal request is
            # byte-identical to what it was before this layer existed.
            return result

        # Compression rewrites message content, so it works on copies: the
        # caller's conversation history (the agent's own state.messages) must
        # survive intact. Only this request is shortened.
        result.messages = [dict(message) for message in messages]

        self._reduce(result)

        self._trace(result)

        return result

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    def _reduce(self, result: ContextBudgetResult) -> None:
        """Bring an oversized context inside the budget, in priority order."""

        units = self._partition(result.messages)

        # Pass 1: shorten the content of old tool results, keeping their
        # messages (and tool_call_ids) so the request stays well-formed.
        result.compression_passes += 1
        result.entries_compressed += self._compress_units(units, kinds=("tool",))

        if self._fits(units):
            self._finish(result, units)
            return

        # Pass 2: shorten old assistant reasoning too.
        result.compression_passes += 1
        result.entries_compressed += self._compress_units(units, kinds=("assistant",))

        if self._fits(units):
            self._finish(result, units)
            return

        # Pass 3: drop whole old exchanges, oldest first, replaced by a
        # digest that states how much was removed.
        result.compression_passes += 1
        result.entries_dropped += self._drop_units(units)

        if not self._fits(units):
            # Hard cap: keep only what must survive, and shrink even that.
            result.hard_capped = True
            self._hard_cap(units)

        self._finish(result, units)

    def _partition(self, messages: list[dict[str, Any]]) -> list[ContextUnit]:
        """Group messages into units and mark the ones that must survive.

        Protected: every system message, the first user message (the original
        task), and the most recent exchanges.
        """

        units: list[ContextUnit] = []

        index = 0
        seen_user = False

        while index < len(messages):
            message = messages[index]
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                group = [message]

                index += 1

                while index < len(messages) and messages[index].get("role") == "tool":
                    group.append(messages[index])
                    index += 1

                units.append(ContextUnit(messages=group, kind="exchange"))
                continue

            protected = role == "system"

            if role == "user" and not seen_user:
                # The original task always survives.
                protected = True
                seen_user = True

            units.append(
                ContextUnit(
                    messages=[message],
                    protected=protected,
                    kind=str(role or "message"),
                )
            )

            index += 1

        # The newest exchanges are protected too: recent tool results are the
        # ones the model is actually working from.
        recent = 0

        for unit in reversed(units):
            if recent >= self.min_recent_messages:
                break

            unit.protected = True
            recent += len(unit.messages)

        return units

    def _compress_units(self, units: list[ContextUnit], kinds: tuple[str, ...]) -> int:
        """Shorten unprotected message content of the given roles."""

        compressed = 0

        for unit in units:
            if unit.protected:
                continue

            for message in unit.messages:
                if message.get("role") not in kinds:
                    continue

                content = str(message.get("content") or "")

                if len(content) <= self.max_entry_chars:
                    continue

                # Role is never changed: a compressed tool result is still a
                # tool result.
                message["content"] = compress_text(content, self.max_entry_chars)

                compressed += 1

        return compressed

    def _drop_units(self, units: list[ContextUnit]) -> int:
        """Drop whole unprotected exchanges, oldest first, until it fits."""

        dropped = 0
        dropped_messages = 0
        dropped_chars = 0

        for position, unit in enumerate(units):
            if self._fits(units):
                break

            if unit.protected or unit.kind == "digest":
                continue

            dropped += 1
            dropped_messages += len(unit.messages)
            dropped_chars += unit.chars

            units[position] = ContextUnit(
                messages=[],
                kind="dropped",
            )

        if dropped:
            self._insert_digest(units, dropped_messages, dropped_chars)

        return dropped

    def _insert_digest(
        self,
        units: list[ContextUnit],
        dropped_messages: int,
        dropped_chars: int,
    ) -> None:
        """Announce what was removed, in counts only.

        The digest is generated text containing no content from the messages
        it replaces, so nothing from a tool result can reach a system role
        through it.
        """

        digest = {
            "role": "system",
            "content": (
                f"{DIGEST_HEADER}\n"
                f"{dropped_messages} earlier messages were removed to stay "
                f"within the context budget.\n"
                f"Removed size: {dropped_chars:,} characters "
                f"(~{estimate_tokens('x' * dropped_chars):,} estimated tokens).\n"
                "Their content is not available. Re-run a search or read if "
                "you need those details again."
            ),
        }

        # Placed after the system instructions so the conversation still opens
        # with NanoCode's own identity.
        insert_at = 0

        for position, unit in enumerate(units):
            if unit.kind == "system":
                insert_at = position + 1

        units.insert(
            insert_at,
            ContextUnit(messages=[digest], protected=True, kind="digest"),
        )

    def _hard_cap(self, units: list[ContextUnit]) -> None:
        """Last resort: shrink even protected content, keeping roles intact.

        Raises ContextBudgetError if the required core still cannot fit, so
        the failure happens here rather than as a provider error.
        """

        # Everything unprotected goes.
        for position, unit in enumerate(units):
            if not unit.protected:
                units[position] = ContextUnit(messages=[], kind="dropped")

        if self._fits(units):
            return

        # Then shrink protected content, newest last so the task and the most
        # recent state survive longest.
        budget_chars = self.max_tokens * CHARS_PER_TOKEN

        for unit in units:
            if self._fits(units):
                return

            for message in unit.messages:
                if message.get("role") == "system" and DIGEST_HEADER in str(
                    message.get("content", "")
                ):
                    continue

                content = str(message.get("content") or "")

                if len(content) <= self.max_entry_chars:
                    continue

                message["content"] = compress_text(content, self.max_entry_chars)

        if self._fits(units):
            return

        required = context_chars(self._flatten(units))

        raise ContextBudgetError(
            "Required context does not fit the configured budget: "
            f"budget {self.max_tokens:,} estimated tokens "
            f"({budget_chars:,} characters), required "
            f"{estimate_tokens('x' * required):,} estimated tokens "
            f"({required:,} characters). "
            "System instructions, the original task and the most recent "
            "exchange could not all be preserved."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten(self, units: list[ContextUnit]) -> list[dict[str, Any]]:
        return [message for unit in units for message in unit.messages]

    def _fits(self, units: list[ContextUnit]) -> bool:
        return (
            estimate_tokens("x" * sum(unit.chars for unit in units))
            <= self.max_tokens
        )

    def _finish(self, result: ContextBudgetResult, units: list[ContextUnit]) -> None:
        messages = self._flatten(units)

        result.messages = messages
        result.compressed = True
        result.final_chars = context_chars(messages)
        result.final_estimated_tokens = estimate_tokens("x" * result.final_chars)

    def _trace(self, result: ContextBudgetResult) -> None:
        """Record what happened. Counts and sizes only -- never content."""

        if self.tracer is None or not result.compressed:
            return

        self.tracer.record(
            "context.compressed",
            component="context",
            **result.to_dict(),
        )

        if result.hard_capped:
            self.tracer.record(
                "context.budget.exceeded",
                component="context",
                budget_tokens=self.max_tokens,
                final_estimated_tokens=result.final_estimated_tokens,
                entries_dropped=result.entries_dropped,
            )
