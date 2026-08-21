"""Recursive self-improvement state for a NanoCode run.

NanoCode already evaluates, reflects and retries. What was missing was an
explicit, inspectable record of *what attempt N+1 learned from attempt N*: the
retry carried only a diagnosis and an improvement line, and nothing tied them
to the attempt number or to the evaluation that produced them.

`RSIContext` is that record. It is deliberately small -- one frozen snapshot
per attempt -- and it produces the planner context in the shape the existing
Planner already accepts, so no pipeline stage had to change shape.

It never invents content. If reflection produced nothing useful, the context
says so (`has_improvement` is False) rather than manufacturing guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# How much of the previous attempt's answer is carried into the next attempt.
# Enough to show what was produced, bounded so a long answer cannot dominate
# the next planning prompt.
MAX_PREVIOUS_RESPONSE_CHARS = 500


def _text(value: Any) -> str:
    """A stripped string for any attribute that may be missing or None."""

    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class RSIContext:
    """What one attempt knows about the attempt before it.

    Attempt numbers are 1-based: attempt 1 is the initial try and carries no
    previous evaluation or reflection.
    """

    attempt: int = 1
    previous_evaluation: Any | None = None
    previous_reflection: Any | None = None
    previous_response: str = ""

    @property
    def is_retry(self) -> bool:
        """True when this attempt exists because an earlier one failed."""

        return self.attempt > 1

    @property
    def has_improvement(self) -> bool:
        """True when reflection actually produced usable guidance."""

        reflection = self.previous_reflection

        if reflection is None:
            return False

        if not getattr(reflection, "should_improve", False):
            return False

        return bool(
            _text(getattr(reflection, "diagnosis", ""))
            or _text(getattr(reflection, "improvement", ""))
        )

    def next_attempt(
        self,
        evaluation: Any,
        reflection: Any,
        response: str = "",
    ) -> "RSIContext":
        """The context the following attempt runs with."""

        return RSIContext(
            attempt=self.attempt + 1,
            previous_evaluation=evaluation,
            previous_reflection=reflection,
            previous_response=response or "",
        )

    def to_planner_context(self) -> dict[str, str] | None:
        """Render this context in the shape Planner.run() already accepts.

        The first attempt has nothing to learn from, so it gets None -- the
        same value the planner received before RSI existed.
        """

        if not self.is_retry:
            return None

        reflection = self.previous_reflection
        evaluation = self.previous_evaluation

        context = {
            # `attempt` and `evaluation` are what make a retry prompt provably
            # different from the attempt before it, even when reflection had
            # nothing to add.
            "attempt": str(self.attempt),
            "diagnosis": _text(getattr(reflection, "diagnosis", "")),
            "improvement": _text(getattr(reflection, "improvement", "")),
        }

        evaluation_reason = _text(getattr(evaluation, "reason", ""))

        if evaluation_reason:
            context["evaluation"] = evaluation_reason

        previous_response = _text(self.previous_response)

        if previous_response:
            context["previous_response"] = previous_response[
                :MAX_PREVIOUS_RESPONSE_CHARS
            ]

        return context

    def summary(self) -> dict[str, Any]:
        """Trace-safe description: counts and flags, no prompt text."""

        return {
            "attempt": self.attempt,
            "is_retry": self.is_retry,
            "has_improvement": self.has_improvement,
        }
