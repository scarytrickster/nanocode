"""Deterministic decomposition of a task into focused RLM child tasks.

Layering: the router decides *whether* a task takes the RLM path; it never
decides what the children are. Decomposition belongs here, in the RLM
orchestration layer, so the two concerns can evolve independently.

This phase is deliberately deterministic — no LLM, no network. A task is
matched against a small ordered list of strategies, each of which contributes
a handful of distinct perspectives on the same subject. A task that matches
nothing is not fanned out at all: it becomes a single child, because splitting
a small task into near-identical pieces wastes budget and produces redundant
children.

`RLMDecomposer` is the interface. Replacing `DeterministicRLMDecomposer` with
an LLM-backed implementation later requires no change to RLMRuntime,
NanoCodeCallHandler, RLMSynthesizer, or RLMRouter.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from rlm.context import RLMContext


class RLMChildTask(NamedTuple):
    """One child task plus the content handed down with it.

    A NamedTuple so it satisfies the `(task, content)` pair contract that
    RLMRuntime.call_many already consumes.
    """

    task: str
    content: str


@dataclass(frozen=True)
class DecompositionStrategy:
    """A named way of splitting a task, selected by trigger phrases.

    Adding a strategy is adding one entry to `STRATEGIES`; the perspectives
    are format strings over the task's subject.
    """

    name: str
    triggers: tuple[str, ...]
    perspectives: tuple[str, ...]

    def matches(self, normalized_task: str) -> bool:
        return any(
            _contains_phrase(phrase, normalized_task)
            for phrase in self.triggers
        )


# ---------------------------------------------------------------------------
# Strategies, in priority order. The first match wins, so a bug hunt that also
# mentions the whole project is investigated rather than surveyed.
# ---------------------------------------------------------------------------

INVESTIGATION_STRATEGY = DecompositionStrategy(
    name="investigation",
    triggers=(
        "bug",
        "root cause",
        "investigate",
        "diagnose",
        "troubleshoot",
        "debug",
        "failing",
        "fails",
        "broken",
        "regression",
        "why",
    ),
    perspectives=(
        "Analyze the core implementation and logic involved in: {subject}",
        "Analyze the configuration, constants, and environment involved in: {subject}",
        "Inspect the call sites, tests, and recent changes related to: {subject}",
    ),
)

COMPARISON_STRATEGY = DecompositionStrategy(
    name="comparison",
    triggers=(
        "compare",
        "comparison",
        "contrast",
        "difference between",
        "differences between",
        "versus",
        "vs",
    ),
    perspectives=(
        "Analyze each implementation involved in: {subject}",
        "Identify the behavioral and structural differences in: {subject}",
    ),
)

SURVEY_STRATEGY = DecompositionStrategy(
    name="survey",
    triggers=(
        "entire project",
        "whole project",
        "entire codebase",
        "whole codebase",
        "entire repository",
        "codebase",
        "across the project",
        "across the codebase",
        "across the repo",
        "project-wide",
        "all files",
        "every file",
        "multiple files",
        "several files",
        "multiple modules",
        "entire system",
        "whole system",
    ),
    perspectives=(
        "Map the modules and source files relevant to: {subject}",
        "Analyze the configuration, dependencies, and entry points relevant to: {subject}",
        "Review the tests and documentation relevant to: {subject}",
    ),
)

STRATEGIES: tuple[DecompositionStrategy, ...] = (
    INVESTIGATION_STRATEGY,
    COMPARISON_STRATEGY,
    SURVEY_STRATEGY,
)

# Used when no strategy matches: one focused child, no redundant fan-out.
SINGLE_CHILD_STRATEGY = DecompositionStrategy(
    name="single",
    triggers=(),
    perspectives=("{subject}",),
)


def _contains_phrase(phrase: str, task: str) -> bool:
    """Case-insensitive word-boundary phrase match, as the router does."""

    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"

    return re.search(pattern, task) is not None


def _subject(task: str) -> str:
    """The task reduced to the subject the perspectives are phrased around."""

    return re.sub(r"\s+", " ", task).strip().rstrip(".").strip()


class RLMDecomposer(ABC):
    """Interface for turning a task into child tasks.

    A decomposer never executes anything: it returns descriptions, and the
    runtime remains the only thing that runs children.
    """

    @abstractmethod
    def decompose(
        self,
        task: str,
        context: RLMContext | None = None,
    ) -> list[RLMChildTask]:
        """Return the child tasks for `task`."""

        raise NotImplementedError


class DeterministicRLMDecomposer(RLMDecomposer):
    """Strategy-based decomposer. Same input, same children, every time."""

    def __init__(
        self,
        strategies: tuple[DecompositionStrategy, ...] | None = None,
        fallback: DecompositionStrategy = SINGLE_CHILD_STRATEGY,
    ) -> None:
        self.strategies = STRATEGIES if strategies is None else tuple(strategies)
        self.fallback = fallback

    def select_strategy(self, task: str) -> DecompositionStrategy:
        """The strategy this task decomposes under."""

        normalized = (task or "").strip().lower()

        if not normalized:
            return self.fallback

        for strategy in self.strategies:
            if strategy.matches(normalized):
                return strategy

        return self.fallback

    def decompose(
        self,
        task: str,
        context: RLMContext | None = None,
    ) -> list[RLMChildTask]:
        """Return focused, mutually distinct child tasks for `task`.

        The original task travels down as each child's `content`, so a child
        can see the goal it is contributing to without the parent having to
        restate it. `context` is accepted for parity with future decomposers
        that will use the surrounding RLM state; the deterministic one depends
        only on the task text.
        """

        subject = _subject(task or "")

        if not subject:
            # Nothing to decompose. The orchestrator turns this into the
            # existing empty-result failure rather than executing anything.
            return []

        strategy = self.select_strategy(task)

        children: list[RLMChildTask] = []
        seen: set[str] = set()

        for perspective in strategy.perspectives:
            child_task = perspective.format(subject=subject).strip()

            # Distinctness guard: two perspectives that collapse to the same
            # request would spend budget twice for one answer.
            fingerprint = re.sub(r"\s+", " ", child_task).lower()

            if not child_task or fingerprint in seen:
                continue

            seen.add(fingerprint)
            children.append(RLMChildTask(task=child_task, content=task))

        return children
