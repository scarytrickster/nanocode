"""Deterministic correctness scoring.

Correctness is not string equality against a golden answer: an answer counts
as correct when it identifies the defect the fixture actually contains -- the
file, the symbol, and an observable trace of the behaviour. Every component of
the score is a transparent boolean or ratio; there is no learned weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.tasks import ExpectedFinding

# Weights for the composite score. Stated here so the number is readable.
CORRECTNESS_WEIGHT = 0.7
COMPLETENESS_WEIGHT = 0.3


@dataclass(frozen=True)
class BenchmarkScore:
    """How well one answer matched what the fixture actually contains."""

    correct: bool = False
    evidence_found: bool = False
    primary_finding_correct: bool = False
    matched_files: tuple[str, ...] = ()
    matched_symbols: tuple[str, ...] = ()
    matched_behaviors: tuple[str, ...] = ()
    completeness: float = 0.0
    score: float = 0.0
    false_positive: bool = False

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "evidence_found": self.evidence_found,
            "primary_finding_correct": self.primary_finding_correct,
            "matched_files": list(self.matched_files),
            "matched_symbols": list(self.matched_symbols),
            "matched_behaviors": list(self.matched_behaviors),
            "completeness": self.completeness,
            "score": self.score,
            "false_positive": self.false_positive,
        }


def _matches(text: str, candidates: tuple[str, ...]) -> tuple[str, ...]:
    lowered = (text or "").lower()

    return tuple(
        candidate
        for candidate in candidates
        if candidate.lower() in lowered
    )


def _claims_a_defect(answer: str) -> bool:
    """Whether the answer asserts a defect at all."""

    lowered = (answer or "").lower()

    if not lowered.strip():
        return False

    denials = ("no defect", "no bug", "no issue", "no relevant source")

    if any(denial in lowered for denial in denials):
        return False

    return True


def score_answer(
    answer: str,
    expected: ExpectedFinding,
    primary_finding: str = "",
    evidence_count: int = 0,
) -> BenchmarkScore:
    """Score one answer against the fixture's known defect.

    `primary_finding` and `evidence_count` come from the RLM evidence
    metadata; a normal run simply leaves them empty, and the parts of the
    score that depend on them stay False/0.
    """

    if not expected.should_find_defect:
        # The fixture has no defect: reporting one is a false positive.
        claimed = _claims_a_defect(answer)

        return BenchmarkScore(
            correct=not claimed,
            evidence_found=evidence_count > 0,
            primary_finding_correct=not claimed,
            completeness=0.0 if claimed else 1.0,
            score=0.0 if claimed else 1.0,
            false_positive=claimed,
        )

    files = _matches(answer, expected.files)
    symbols = _matches(answer, expected.symbols)
    behaviors = _matches(answer, expected.behaviors)

    satisfied = sum(
        1
        for required, found in (
            (expected.files, files),
            (expected.symbols, symbols),
            (expected.behaviors, behaviors),
        )
        if required and found
    )

    completeness = (
        satisfied / expected.criteria_count if expected.criteria_count else 0.0
    )

    # Correct means the answer located the defect: the right file, the right
    # symbol, and at least one observable sign of the behaviour.
    correct = bool(
        (not expected.files or files)
        and (not expected.symbols or symbols)
        and (not expected.behaviors or behaviors)
    )

    primary_correct = bool(
        primary_finding
        and (not expected.files or _matches(primary_finding, expected.files))
        and (not expected.symbols or _matches(primary_finding, expected.symbols))
    )

    score = round(
        CORRECTNESS_WEIGHT * (1.0 if correct else 0.0)
        + COMPLETENESS_WEIGHT * completeness,
        4,
    )

    return BenchmarkScore(
        correct=correct,
        evidence_found=evidence_count > 0,
        primary_finding_correct=primary_correct,
        matched_files=files,
        matched_symbols=symbols,
        matched_behaviors=behaviors,
        completeness=round(completeness, 4),
        score=score,
        false_positive=False,
    )
