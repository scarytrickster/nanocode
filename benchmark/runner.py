"""Runs benchmark tasks down the normal and the RLM path.

Both runs use the same task, the same fixture and the same scripted model.
Only the execution path differs, so the comparison is between the two paths
rather than between two setups.

The router is never bypassed silently: its real decision for the prompt is
recorded on the result, and the two paths are then both forced so they can be
compared on the same task.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import config.settings as settings
from agent.agent import NanoCodeAgent
from benchmark.fakes import ScriptedModel
from benchmark.fixtures import FIXTURES
from benchmark.metrics import ExecutionMetrics, collect_metrics
from benchmark.scoring import BenchmarkScore, score_answer
from benchmark.tasks import BenchmarkTask
from rlm.budget import RLMBudget
from rlm.nanocode_handler import NanoCodeCallHandler
from rlm.orchestrator import RLMOrchestrator
from rlm.router import STRATEGY_NORMAL, STRATEGY_RLM, RLMRouter, RouteDecision
from rlm.runtime import DEFAULT_MAX_CONCURRENCY, RLMRuntime
from rlm.synthesizer import (
    CONFIDENCE_KEY,
    CONFIRMED_COUNT_KEY,
    CONFLICTING_COUNT_KEY,
    FINDINGS_KEY,
    PARTIAL_KEY,
    PRIMARY_KEY,
)
from rlm.evidence import CONFIRMED, CONFLICTING, LIKELY, POSSIBLE, UNSUPPORTED


@dataclass
class EvidenceMetrics:
    """Phase 8 synthesis metrics, read from the public result metadata."""

    findings_count: int = 0
    confirmed_findings: int = 0
    likely_findings: int = 0
    possible_findings: int = 0
    unsupported_findings: int = 0
    conflicting_findings: int = 0
    evidence_items: int = 0
    primary_finding: str = ""
    confidence: str = ""
    partial: bool = False

    def to_dict(self) -> dict:
        return {
            "findings_count": self.findings_count,
            "confirmed_findings": self.confirmed_findings,
            "likely_findings": self.likely_findings,
            "possible_findings": self.possible_findings,
            "unsupported_findings": self.unsupported_findings,
            "conflicting_findings": self.conflicting_findings,
            "evidence_items": self.evidence_items,
            "primary_finding": self.primary_finding,
            "confidence": self.confidence,
            "partial": self.partial,
        }


@dataclass
class PathResult:
    """One task executed down one path."""

    strategy: str
    success: bool
    answer: str
    metrics: ExecutionMetrics
    score: BenchmarkScore
    evidence: EvidenceMetrics = field(default_factory=EvidenceMetrics)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "success": self.success,
            "answer": self.answer,
            "metrics": self.metrics.to_dict(),
            "score": self.score.to_dict(),
            "evidence": self.evidence.to_dict(),
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    """A task's outcome on both paths, plus what the router would have chosen."""

    task: BenchmarkTask
    normal: PathResult
    rlm: PathResult
    router_strategy: str = STRATEGY_NORMAL
    router_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task.id,
            "category": self.task.category,
            "router_strategy": self.router_strategy,
            "normal": self.normal.to_dict(),
            "rlm": self.rlm.to_dict(),
        }


def evidence_from(metadata: dict) -> EvidenceMetrics:
    """Read Phase 8 metrics from the public synthesis metadata."""

    findings = metadata.get(FINDINGS_KEY, []) or []

    def count(status: str) -> int:
        return sum(1 for finding in findings if finding.get("status") == status)

    return EvidenceMetrics(
        findings_count=len(findings),
        confirmed_findings=metadata.get(CONFIRMED_COUNT_KEY, count(CONFIRMED)),
        likely_findings=count(LIKELY),
        possible_findings=count(POSSIBLE),
        unsupported_findings=count(UNSUPPORTED),
        conflicting_findings=metadata.get(
            CONFLICTING_COUNT_KEY, count(CONFLICTING)
        ),
        evidence_items=sum(
            len(finding.get("evidence", []) or []) for finding in findings
        ),
        primary_finding=str(metadata.get(PRIMARY_KEY, "")),
        confidence=str(metadata.get(CONFIDENCE_KEY, "")),
        partial=bool(metadata.get(PARTIAL_KEY, False)),
    )


class FixedRouter:
    """Forces one path so both can be measured on the same task.

    The real router's opinion is recorded separately; this only removes the
    branch so the two paths are comparable.
    """

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def decide(self, task: str) -> RouteDecision:
        return RouteDecision(
            strategy=self.strategy,
            reason="benchmark: path forced for comparison",
            score=1.0 if self.strategy == STRATEGY_RLM else 0.0,
            signals=[],
        )


@contextlib.contextmanager
def scripted_model(project: str):
    """Replace only the model call, leaving the whole stack above it real."""

    model = ScriptedModel(project=project)

    original = settings.client.chat.completions.create

    settings.client.chat.completions.create = model.create

    try:
        yield model
    finally:
        settings.client.chat.completions.create = original


def _run(
    task: BenchmarkTask,
    project: str,
    strategy: str,
    clock: Callable[[], float] = perf_counter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> PathResult:
    """Execute one task down one path and measure it."""

    events: list = []

    with scripted_model(project) as model:
        agent = NanoCodeAgent(
            console_trace=False,
            trace_callback=events.append,
            router=FixedRouter(strategy),
        )

        if strategy == STRATEGY_RLM:
            # The real orchestrator, runtime, handler, decomposer and
            # synthesizer; only the model underneath the children is scripted.
            agent.rlm_orchestrator = RLMOrchestrator(
                runtime=RLMRuntime(
                    call_handler=NanoCodeCallHandler(
                        agent_factory=lambda: NanoCodeAgent(
                            console_trace=False,
                            rlm_enabled=False,
                        ),
                    ),
                    budget=RLMBudget(
                        max_depth=2,
                        max_children=4,
                        max_iterations=8,
                    ),
                    max_concurrency=max_concurrency,
                ),
                tracer=agent.tracer,
            )

        started = clock()
        error = ""
        answer = ""

        try:
            answer = agent.run(task.prompt)
        except Exception as failure:  # a crashed run is a recorded outcome
            error = f"{type(failure).__name__}: {failure}"

        duration = max(0.0, clock() - started)

    metrics = collect_metrics(events, duration_seconds=duration)

    runtime = getattr(agent.rlm_orchestrator, "runtime", None)

    if strategy == STRATEGY_RLM and runtime is not None:
        # Configured limit and what actually overlapped, straight from the
        # runtime's own counters.
        metrics.max_concurrency = runtime.max_concurrency
        metrics.peak_active_children = runtime.peak_active_children

    evidence = EvidenceMetrics()

    if strategy == STRATEGY_RLM and agent.rlm_orchestrator is not None:
        result = agent.rlm_orchestrator.last_result

        if result is not None:
            evidence = evidence_from(result.metadata or {})

    score = score_answer(
        answer=answer,
        expected=task.expected,
        primary_finding=evidence.primary_finding,
        evidence_count=evidence.evidence_items,
    )

    success = bool(answer) and not error

    if error:
        metrics.failures.append(error)

    return PathResult(
        strategy=strategy,
        success=success,
        answer=answer,
        metrics=metrics,
        score=score,
        evidence=evidence,
        error=error,
    )


def run_task(
    task: BenchmarkTask,
    root: str,
    clock: Callable[[], float] = perf_counter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> BenchmarkResult:
    """Run one benchmark task down both paths against a fresh fixture."""

    fixture = FIXTURES[task.fixture]

    project = fixture.write(os.path.join(root, task.id))

    decision = RLMRouter().decide(task.prompt)

    return BenchmarkResult(
        task=task,
        normal=_run(task, project, STRATEGY_NORMAL, clock),
        rlm=_run(task, project, STRATEGY_RLM, clock, max_concurrency),
        router_strategy=decision.strategy,
        router_reason=decision.reason,
    )


def run_benchmark(
    tasks,
    root: str,
    clock: Callable[[], float] = perf_counter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> list[BenchmarkResult]:
    """Run every task. Failures are recorded, never skipped."""

    return [run_task(task, root, clock, max_concurrency) for task in tasks]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    values = list(values)

    return round(sum(values) / len(values), 3) if values else 0.0


def aggregate(results: list[BenchmarkResult]) -> dict:
    """Aggregate metrics across tasks. An empty run aggregates to zeros."""

    total = len(results)

    if not total:
        return {
            "tasks": 0,
            "normal": {},
            "rlm": {},
        }

    def side(selector) -> dict:
        runs = [selector(result) for result in results]

        return {
            "correct": sum(1 for run in runs if run.score.correct),
            "correct_rate": round(
                sum(1 for run in runs if run.score.correct) / total, 3
            ),
            "success": sum(1 for run in runs if run.success),
            "success_rate": round(
                sum(1 for run in runs if run.success) / total, 3
            ),
            "false_positives": sum(
                1 for run in runs if run.score.false_positive
            ),
            "average_score": _mean(run.score.score for run in runs),
            "average_duration": _mean(
                run.metrics.duration_seconds for run in runs
            ),
            "average_llm_calls": _mean(run.metrics.llm_calls for run in runs),
            "average_tool_calls": _mean(run.metrics.tool_calls for run in runs),
            "average_children": _mean(run.metrics.children for run in runs),
            "peak_active_children": max(
                (run.metrics.peak_active_children for run in runs), default=0
            ),
            "average_attempts": _mean(run.metrics.attempts for run in runs),
            "average_findings": _mean(
                run.evidence.findings_count for run in runs
            ),
            "failures": sum(run.metrics.failure_count for run in runs),
            "partial_rate": round(
                sum(1 for run in runs if run.evidence.partial) / total, 3
            ),
        }

    return {
        "tasks": total,
        "normal": side(lambda result: result.normal),
        "rlm": side(lambda result: result.rlm),
    }
