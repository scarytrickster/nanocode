"""Execution metrics derived from the events NanoCode already emits.

No second instrumentation system: every number here is counted from real
TraceEvent objects recorded by the existing tracer. Because RLM child agents
forward their events into the parent tracer, one sink on the parent captures
both parent and child activity.

Token usage is deliberately absent. The existing instrumentation does not
expose it deterministically, and inventing a number would be worse than
reporting that it is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Events that mean "one model request was issued".
PLANNER_CALL = "planner.started"
EXECUTOR_CALL = "llm.started"

TOOL_STARTED = "tool.started"
TOOL_FAILED = "tool.failed"

CHILD_STARTED = "rlm.child.started"
CHILD_COMPLETED = "rlm.child.completed"
CHILD_FAILED = "rlm.child.failed"
CHILD_RETRYING = "rlm.child.retrying"

RETRY_STARTED = "retry.started"
RETRY_EXHAUSTED = "retry.exhausted"
RSI_STARTED = "rsi.started"
RSI_RETRY = "rsi.retry.started"
RSI_EXHAUSTED = "rsi.exhausted"

TOKENS_UNAVAILABLE = "unavailable: not exposed by the current instrumentation"


@dataclass
class ExecutionMetrics:
    """What one run actually did."""

    llm_calls: int = 0
    planner_calls: int = 0
    executor_calls: int = 0

    tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)

    attempts: int = 1
    rsi_retries: int = 0
    rsi_used_improvement: bool = False
    retry_exhausted: bool = False

    children: int = 0
    successful_children: int = 0
    failed_children: int = 0
    child_retries: int = 0

    failures: list[str] = field(default_factory=list)

    max_concurrency: int = 1
    peak_active_children: int = 0

    duration_seconds: float = 0.0
    tokens: str = TOKENS_UNAVAILABLE

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "planner_calls": self.planner_calls,
            "executor_calls": self.executor_calls,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": dict(self.tool_calls_by_name),
            "attempts": self.attempts,
            "rsi_retries": self.rsi_retries,
            "rsi_used_improvement": self.rsi_used_improvement,
            "retry_exhausted": self.retry_exhausted,
            "children": self.children,
            "successful_children": self.successful_children,
            "failed_children": self.failed_children,
            "child_retries": self.child_retries,
            "max_concurrency": self.max_concurrency,
            "peak_active_children": self.peak_active_children,
            "failures": list(self.failures),
            "duration_seconds": self.duration_seconds,
            "tokens": self.tokens,
        }


def collect_metrics(events, duration_seconds: float = 0.0) -> ExecutionMetrics:
    """Count what happened from the recorded trace events.

    Only events that represent real work are counted: a tool named in a prompt
    is invisible here, because nothing recorded a `tool.started` for it.
    """

    metrics = ExecutionMetrics(duration_seconds=duration_seconds)

    for event in events:
        name = event.name

        if name == PLANNER_CALL:
            metrics.planner_calls += 1

        elif name == EXECUTOR_CALL:
            metrics.executor_calls += 1

        elif name == TOOL_STARTED:
            tool = str(event.data.get("tool", "unknown"))

            metrics.tool_calls += 1
            metrics.tool_calls_by_name[tool] = (
                metrics.tool_calls_by_name.get(tool, 0) + 1
            )

        elif name == CHILD_STARTED:
            metrics.children += 1

        elif name == CHILD_COMPLETED:
            metrics.successful_children += 1

        elif name == CHILD_FAILED:
            metrics.failed_children += 1

            metrics.failures.append(
                f"child {event.data.get('child', '?')}: "
                f"{event.data.get('error_type', 'error')}"
            )

        elif name == CHILD_RETRYING:
            metrics.child_retries += 1

        elif name == RETRY_STARTED:
            metrics.attempts += 1

        elif name == RETRY_EXHAUSTED:
            metrics.retry_exhausted = True

        elif name == RSI_RETRY:
            metrics.rsi_retries += 1

            if event.data.get("has_improvement"):
                metrics.rsi_used_improvement = True

        elif name.endswith(".failed") and not name.startswith("rlm."):
            metrics.failures.append(f"{event.component}: {name}")

    metrics.llm_calls = metrics.planner_calls + metrics.executor_calls

    return metrics
