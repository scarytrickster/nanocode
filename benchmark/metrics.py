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

DECOMPOSITION_STARTED = "rlm.decomposition.started"
DECOMPOSITION_SUCCEEDED = "rlm.decomposition.succeeded"
DECOMPOSITION_FALLBACK = "rlm.decomposition.fallback"

CONTEXT_COMPRESSED = "context.compressed"

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

    context_compressions: int = 0
    context_original_tokens: int = 0
    context_final_tokens: int = 0
    peak_context_tokens: int = 0

    decomposition_strategy: str = ""
    decomposition_llm_calls: int = 0
    decomposition_fallback: bool = False
    decomposition_validation_failures: int = 0

    duration_seconds: float = 0.0
    tokens: str = TOKENS_UNAVAILABLE

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def context_compression_ratio(self) -> float:
        """Final size as a fraction of the original. 1.0 when uncompressed."""

        if not self.context_original_tokens:
            return 1.0

        return round(
            self.context_final_tokens / self.context_original_tokens, 4
        )

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
            "context_compressions": self.context_compressions,
            "context_original_tokens": self.context_original_tokens,
            "context_final_tokens": self.context_final_tokens,
            "peak_context_tokens": self.peak_context_tokens,
            "context_compression_ratio": self.context_compression_ratio,
            "decomposition_strategy": self.decomposition_strategy,
            "decomposition_llm_calls": self.decomposition_llm_calls,
            "decomposition_fallback": self.decomposition_fallback,
            "decomposition_validation_failures": (
                self.decomposition_validation_failures
            ),
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

        elif name == CONTEXT_COMPRESSED:
            metrics.context_compressions += 1

            original = int(event.data.get("original_estimated_tokens", 0))
            final = int(event.data.get("final_estimated_tokens", 0))

            # Totals across every compressed request, plus the largest single
            # context the run ever assembled.
            metrics.context_original_tokens += original
            metrics.context_final_tokens += final
            metrics.peak_context_tokens = max(
                metrics.peak_context_tokens, original
            )

        elif name == DECOMPOSITION_STARTED:
            metrics.decomposition_strategy = str(
                event.data.get("strategy", "")
            )

        elif name == DECOMPOSITION_SUCCEEDED:
            metrics.decomposition_strategy = str(
                event.data.get("strategy", metrics.decomposition_strategy)
            )
            metrics.decomposition_validation_failures = int(
                event.data.get("validation_failures", 0)
            )

            # An LLM decomposition costs exactly one model request; a
            # deterministic one costs none.
            if event.data.get("strategy") == "llm":
                metrics.decomposition_llm_calls = 1

        elif name == DECOMPOSITION_FALLBACK:
            metrics.decomposition_fallback = True
            metrics.decomposition_strategy = "deterministic (fallback)"
            metrics.decomposition_validation_failures = int(
                event.data.get("validation_failures", 0)
            )

            # The attempt was made and paid for before it failed.
            metrics.decomposition_llm_calls = 1

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
