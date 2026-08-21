"""Rendering for benchmark results.

The report states plainly what it is: an engineering benchmark over a handful
of synthetic fixtures, run against a scripted model. It is not evidence that
one path is better than the other, and it says so.
"""

from __future__ import annotations

from benchmark.runner import BenchmarkResult, aggregate

DISCLAIMER = (
    "Engineering benchmark, not a scientific evaluation: a handful of "
    "synthetic fixtures run against a scripted model. No statistical "
    "significance is claimed."
)

WIDTH = 62


def _mark(value: bool) -> str:
    return "yes" if value else "no"


def _row(label: str, normal, rlm) -> str:
    return f"{label:<18}{str(normal):<14}{rlm}"


def render_result(result: BenchmarkResult) -> str:
    """One task's normal-vs-RLM comparison."""

    normal, rlm = result.normal, result.rlm

    lines = [
        "=" * WIDTH,
        f"Task: {result.task.id}",
        f"Category: {result.task.category}   Difficulty: {result.task.difficulty}",
        f"Router would choose: {result.router_strategy}",
        "-" * WIDTH,
        _row("", "Normal", "RLM"),
        "-" * WIDTH,
        _row("Correct", _mark(normal.score.correct), _mark(rlm.score.correct)),
        _row("Success", _mark(normal.success), _mark(rlm.success)),
        _row("Score", normal.score.score, rlm.score.score),
        _row("Completeness", normal.score.completeness, rlm.score.completeness),
        _row(
            "Duration (s)",
            f"{normal.metrics.duration_seconds:.3f}",
            f"{rlm.metrics.duration_seconds:.3f}",
        ),
        _row("LLM calls", normal.metrics.llm_calls, rlm.metrics.llm_calls),
        _row("Tool calls", normal.metrics.tool_calls, rlm.metrics.tool_calls),
        _row("Attempts", normal.metrics.attempts, rlm.metrics.attempts),
        _row("Children", normal.metrics.children, rlm.metrics.children),
        _row(
            "Children ok",
            normal.metrics.successful_children,
            rlm.metrics.successful_children,
        ),
        _row(
            "Children failed",
            normal.metrics.failed_children,
            rlm.metrics.failed_children,
        ),
        _row(
            "Primary correct",
            "-",
            _mark(rlm.score.primary_finding_correct),
        ),
        _row("Findings", "-", rlm.evidence.findings_count),
        _row("Evidence items", "-", rlm.evidence.evidence_items),
        _row("Confidence", "-", rlm.evidence.confidence or "-"),
        _row("Partial", _mark(normal.evidence.partial), _mark(rlm.evidence.partial)),
        _row(
            "Failures",
            normal.metrics.failure_count,
            rlm.metrics.failure_count,
        ),
        _row("Tokens", normal.metrics.tokens, ""),
    ]

    if normal.metrics.failures or rlm.metrics.failures:
        lines.append("-" * WIDTH)
        lines.append("Recorded failures:")

        for failure in normal.metrics.failures:
            lines.append(f"  normal: {failure}")

        for failure in rlm.metrics.failures:
            lines.append(f"  rlm:    {failure}")

    return "\n".join(lines)


def render_aggregate(results: list[BenchmarkResult]) -> str:
    """Totals across every task."""

    totals = aggregate(results)

    if not totals["tasks"]:
        return "\n".join(
            [
                "=" * WIDTH,
                "NanoCode Benchmark -- aggregate",
                "=" * WIDTH,
                "No tasks were evaluated.",
            ]
        )

    normal, rlm = totals["normal"], totals["rlm"]

    lines = [
        "=" * WIDTH,
        "NanoCode Benchmark -- aggregate",
        "=" * WIDTH,
        f"Tasks evaluated: {totals['tasks']}",
        "-" * WIDTH,
        _row("", "Normal", "RLM"),
        "-" * WIDTH,
        _row("Correct", normal["correct"], rlm["correct"]),
        _row("Correct rate", normal["correct_rate"], rlm["correct_rate"]),
        _row("Success rate", normal["success_rate"], rlm["success_rate"]),
        _row("False positives", normal["false_positives"], rlm["false_positives"]),
        _row("Avg score", normal["average_score"], rlm["average_score"]),
        _row("Avg duration (s)", normal["average_duration"], rlm["average_duration"]),
        _row("Avg LLM calls", normal["average_llm_calls"], rlm["average_llm_calls"]),
        _row("Avg tool calls", normal["average_tool_calls"], rlm["average_tool_calls"]),
        _row("Avg attempts", normal["average_attempts"], rlm["average_attempts"]),
        _row("Avg children", normal["average_children"], rlm["average_children"]),
        _row("Avg findings", normal["average_findings"], rlm["average_findings"]),
        _row("Partial rate", normal["partial_rate"], rlm["partial_rate"]),
        _row("Failures", normal["failures"], rlm["failures"]),
        "-" * WIDTH,
        f"Token usage: {results[0].normal.metrics.tokens}",
        "=" * WIDTH,
        DISCLAIMER,
        "=" * WIDTH,
    ]

    return "\n".join(lines)


def render_report(results: list[BenchmarkResult]) -> str:
    """The full report: per-task comparisons followed by the aggregate."""

    header = [
        "=" * WIDTH,
        "NanoCode Benchmark",
        "=" * WIDTH,
        DISCLAIMER,
        "",
    ]

    body = [render_result(result) for result in results]

    return "\n".join(header + body + ["", render_aggregate(results)])
