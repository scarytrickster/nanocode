from __future__ import annotations

import re
from dataclasses import dataclass, field

# Strategy identifiers returned by the router.
STRATEGY_NORMAL = "normal"
STRATEGY_RLM = "rlm"


@dataclass
class RouteDecision:
    """Decision describing how a task should be executed."""

    strategy: str
    reason: str
    score: float

    signals: list[str] = field(default_factory=list)

    @property
    def is_rlm(self) -> bool:
        return self.strategy == STRATEGY_RLM


# ---------------------------------------------------------------------------
# Heuristic signals.
#
# Each entry is (label, phrase, weight). Phrases are matched case-insensitively
# on word boundaries, so the router is fully deterministic.
#
# The design intent:
#   - breadth / investigation / comparison signals are strong on their own
#   - generic verbs ("analyze", "find") are weak and never trigger RLM alone
#   - small, local edits actively pull the score down
# ---------------------------------------------------------------------------

BREADTH_SIGNALS: list[tuple[str, float]] = [
    ("entire project", 0.6),
    ("whole project", 0.6),
    ("entire codebase", 0.6),
    ("whole codebase", 0.6),
    ("entire repository", 0.6),
    ("across the project", 0.6),
    ("across the codebase", 0.6),
    ("across the repo", 0.6),
    ("project-wide", 0.6),
    ("codebase", 0.6),
    ("multiple files", 0.6),
    ("several files", 0.6),
    ("all files", 0.6),
    ("all the files", 0.6),
    ("every file", 0.6),
    ("each file", 0.6),
    ("these files", 0.6),
    ("multiple modules", 0.6),
    ("these implementations", 0.6),
    ("entire system", 0.6),
    ("whole system", 0.6),
]

INVESTIGATION_SIGNALS: list[tuple[str, float]] = [
    ("root cause", 0.6),
    ("investigate", 0.6),
    ("diagnose", 0.6),
    ("troubleshoot", 0.6),
    ("debug why", 0.6),
    ("why is", 0.4),
    ("why this", 0.4),
    ("why the", 0.4),
    ("is failing", 0.4),
    ("keeps failing", 0.4),
    ("is broken", 0.4),
]

COMPARISON_SIGNALS: list[tuple[str, float]] = [
    ("compare", 0.5),
    ("contrast", 0.5),
    ("difference between", 0.5),
    ("differences between", 0.5),
    ("versus", 0.5),
    ("vs", 0.5),
]

# Weak signals: useful context, but never decisive on their own.
ANALYSIS_SIGNALS: list[tuple[str, float]] = [
    ("analyze", 0.25),
    ("analyse", 0.25),
    ("find", 0.25),
    ("examine", 0.25),
    ("review", 0.25),
    ("audit", 0.25),
    ("explore", 0.25),
    ("trace", 0.25),
    ("understand", 0.25),
]

# Negative signals: small, local, single-step work.
SIMPLE_SIGNALS: list[tuple[str, float]] = [
    ("typo", -0.3),
    ("rename", -0.3),
    ("missing import", -0.3),
    ("add an import", -0.3),
    ("this variable", -0.3),
    ("this line", -0.3),
    ("one-line", -0.3),
    ("what is", -0.3),
    ("what does", -0.3),
    ("how do i", -0.3),
]

SIGNAL_GROUPS: list[tuple[str, list[tuple[str, float]]]] = [
    ("breadth", BREADTH_SIGNALS),
    ("investigation", INVESTIGATION_SIGNALS),
    ("comparison", COMPARISON_SIGNALS),
    ("analysis", ANALYSIS_SIGNALS),
    ("simple", SIMPLE_SIGNALS),
]

# Weak groups are capped so that stacking generic verbs ("find and analyze")
# can never reach the threshold on its own; they only ever nudge a score that
# a strong signal has already earned.
GROUP_CAPS: dict[str, float] = {
    "analysis": 0.25,
}

DEFAULT_THRESHOLD = 0.5


def _matches(phrase: str, task: str) -> bool:
    """Case-insensitive word-boundary phrase match."""

    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"

    return re.search(pattern, task) is not None


class RLMRouter:
    """Deterministic router choosing between normal and RLM execution."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def decide(self, task: str) -> RouteDecision:
        """Decide how a task should be executed."""

        normalized = (task or "").strip().lower()

        if not normalized:
            return RouteDecision(
                strategy=STRATEGY_NORMAL,
                reason="Empty task; defaulting to normal execution.",
                score=0.0,
                signals=[],
            )

        score = 0.0
        signals: list[str] = []

        for group, entries in SIGNAL_GROUPS:
            group_score = 0.0

            for phrase, weight in entries:
                if _matches(phrase, normalized):
                    group_score += weight
                    signals.append(f"{group}:{phrase}")

            cap = GROUP_CAPS.get(group)

            if cap is not None:
                group_score = min(group_score, cap)

            score += group_score

        score = round(min(max(score, 0.0), 1.0), 2)

        if score >= self.threshold:
            strategy = STRATEGY_RLM
            verdict = "complex task"
        else:
            strategy = STRATEGY_NORMAL
            verdict = "simple task"

        matched = ", ".join(signals) if signals else "none"

        reason = (
            f"{verdict}: score {score:.2f} "
            f"({'>=' if score >= self.threshold else '<'} "
            f"threshold {self.threshold:.2f}); "
            f"signals: {matched}"
        )

        return RouteDecision(
            strategy=strategy,
            reason=reason,
            score=score,
            signals=signals,
        )
