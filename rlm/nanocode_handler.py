from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import RateLimitError

from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult
from agent.agent import NanoCodeAgent

# Error classifications recorded in RLMResult.metadata["error_type"].
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_ERROR = "error"

# Bounded application-level retry for rate limits only. This is per child and
# never scales with the number of children: three children retrying once each
# is three extra requests, not nine.
DEFAULT_RATE_LIMIT_RETRIES = 1
DEFAULT_RATE_LIMIT_DELAY = 5.0


def is_rate_limit_error(error: BaseException) -> bool:
    """True only for real provider rate limits, never for other failures."""

    return isinstance(error, RateLimitError)


def classify_error(error: BaseException) -> str:
    """The error_type recorded for a failed child."""

    return (
        ERROR_TYPE_RATE_LIMIT
        if is_rate_limit_error(error)
        else ERROR_TYPE_ERROR
    )


@dataclass(frozen=True)
class NanoCodeRequest:
    """Information passed from an RLM child context to NanoCode."""

    task: str
    content: str
    depth: int
    metadata: dict[str, Any]

def create_nanocode_agent() -> NanoCodeAgent:
    """
    Create a fresh NanoCodeAgent for an RLM child execution.

    Children run the normal NanoCode pipeline: rlm_enabled=False stops a child
    from routing back into the RLM path.
    """

    return NanoCodeAgent(
            console_trace=False,
            rlm_enabled=False,
    )


class NanoCodeCallHandler(RLMCallHandler):
    """
    Adapter between the RLM runtime and NanoCodeAgent.

    The adapter keeps RLM-specific context separate from the existing
    NanoCodeAgent interface.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.agent_factory = agent_factory
        self.last_request: NanoCodeRequest | None = None

        # `sleep` is injectable so tests can prove the retry happened without
        # actually waiting.
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_delay = rate_limit_delay
        self.sleep = sleep

        # Event forwarding. A child agent has its own Tracer, which by default
        # nobody is listening to, so everything the child did was invisible
        # outside Langfuse. When a parent tracer is attached, the child's real
        # events are replayed into it and reach whatever the parent already
        # renders with.
        self.event_tracer: Any | None = None
        self.child_count: int = 0
        self.children_started: int = 0

        # Children may run concurrently, so identity cannot come from the
        # order threads happen to arrive: the runtime announces the
        # decomposition order up front and each child looks itself up.
        self._child_order: dict[str, int] = {}
        self._counter_lock = threading.Lock()

    def set_event_forwarding(
        self,
        tracer: Any | None,
        child_count: int = 0,
    ) -> None:
        """Forward child agent events into `tracer` (the parent's)."""

        self.event_tracer = tracer
        self.child_count = child_count

        with self._counter_lock:
            self.children_started = 0

    def set_child_order(self, tasks: list[str]) -> None:
        """Fix each child's number from the decomposition order."""

        self._child_order = {
            task: index
            for index, task in enumerate(tasks, start=1)
        }

        if not self.child_count:
            self.child_count = len(tasks)

        with self._counter_lock:
            self.children_started = 0

    def _child_index(self, context: RLMContext) -> int:
        """This child's number, stable regardless of when it runs."""

        index = self._child_order.get(context.task)

        if index is not None:
            with self._counter_lock:
                self.children_started += 1

            return index

        # No announced order (a direct runtime.call): fall back to arrival
        # order, counted atomically.
        with self._counter_lock:
            self.children_started += 1

            return self.children_started

    def _forward(
        self,
        event: Any,
        child_index: int,
        task: str,
        depth: int,
    ) -> None:
        """Replay one real child event into the parent tracer.

        Nothing is invented here: the name, component, timing and payload are
        the child's own. Only the child's identity is added.
        """

        tracer = self.event_tracer

        if tracer is None:
            return

        data = dict(getattr(event, "data", {}) or {})

        data.update(
            {
                "rlm_child": child_index,
                "rlm_child_count": self.child_count,
                "rlm_child_task": task,
                "depth": depth,
            }
        )

        tracer.record(
            getattr(event, "name", "event"),
            component=getattr(event, "component", "agent"),
            duration_ms=getattr(event, "duration_ms", None),
            error=getattr(event, "error", None),
            **data,
        )

    def _attach_event_forwarding(
        self,
        agent: Any,
        child_index: int,
        context: RLMContext,
    ) -> None:
        """Listen to a child agent's tracer, if it has one."""

        if self.event_tracer is None:
            return

        tracer = getattr(agent, "tracer", None)

        if tracer is None:
            return

        tracer.on_event = lambda event: self._forward(
            event,
            child_index=child_index,
            task=context.task,
            depth=context.depth,
        )

    def _record(self, name: str, **data: Any) -> None:
        """Record an RLM-level event on the parent tracer."""

        if self.event_tracer is None:
            return

        self.event_tracer.record(name, component="rlm", **data)

    def call(self, context: RLMContext) -> RLMResult:
        """Execute a child task using NanoCodeAgent."""

        child_index = self._child_index(context)

        agent = self.agent_factory()

        self._attach_event_forwarding(agent, child_index, context)

        request = NanoCodeRequest(
            task=context.task,
            content=context.content,
            depth=context.depth,
            metadata=dict(context.metadata),
        )

        self.last_request = request

        self._record(
            "rlm.child.started",
            child=child_index,
            of=self.child_count,
            task=context.task,
            depth=context.depth,
        )

        attempts = 0

        while True:
            attempts += 1

            try:
                # NanoCodeAgent.run() currently accepts a string task.
                response = agent.run(request.task)

            except Exception as error:
                retryable = (
                    is_rate_limit_error(error)
                    and attempts <= self.rate_limit_retries
                )

                if retryable:
                    # Only rate limits are retried: a deterministic failure
                    # would fail again identically and just burn budget.
                    self.sleep(self.rate_limit_delay)

                    self._record(
                        "rlm.child.retrying",
                        child=child_index,
                        of=self.child_count,
                        attempt=attempts,
                    )

                    # A fresh agent per attempt keeps attempts isolated, the
                    # same way each child gets its own agent.
                    agent = self.agent_factory()

                    self._attach_event_forwarding(agent, child_index, context)

                    continue

                # One child failing must not abort its siblings or corrupt the
                # parent context. It becomes an unsuccessful result, which is
                # the failure shape RLMSynthesizer already knows how to handle.
                self._record(
                    "rlm.child.failed",
                    child=child_index,
                    of=self.child_count,
                    error=str(error),
                    error_type=classify_error(error),
                    attempts=attempts,
                )

                return self._to_failure(
                    context=context,
                    error=error,
                    attempts=attempts,
                )

            result = self._to_rlm_result(
                context=context,
                response=response,
                attempts=attempts,
            )

            self._record(
                "rlm.child.completed" if result.success else "rlm.child.failed",
                child=child_index,
                of=self.child_count,
                attempts=attempts,
            )

            return result

    def _to_failure(
        self,
        context: RLMContext,
        error: BaseException,
        attempts: int,
    ) -> RLMResult:
        """Convert a child exception into the existing failed-result shape."""

        error_type = classify_error(error)

        return RLMResult(
            answer="",
            success=False,
            depth=context.depth,
            metadata={
                "error": str(error),
                "error_type": error_type,
                "rate_limited": error_type == ERROR_TYPE_RATE_LIMIT,
                "attempts": attempts,
                "task": context.task,
            },
        )

    def create_nanocode_agent() -> NanoCodeAgent:
        """
        Create a fresh NanoCodeAgent for an RLM child execution.

        A fresh instance prevents child executions from accidentally
        sharing mutable agent state with the parent.
        """

        return NanoCodeAgent(
            console_trace=False,
        )

    def _to_rlm_result(
        self,
        context: RLMContext,
        response: Any,
        attempts: int = 1,
    ) -> RLMResult:
        """Convert a NanoCode response into an RLMResult."""

        if isinstance(response, RLMResult):
            return response

        if isinstance(response, str):
            return RLMResult(
                answer=response,
                success=True,
                depth=context.depth,
                metadata={"attempts": attempts},
            )

        if response is None:
            return RLMResult(
                answer="",
                success=False,
                depth=context.depth,
            )

        answer = getattr(response, "answer", None)

        if answer is not None:
            success = getattr(response, "success", True)

            return RLMResult(
                answer=str(answer),
                success=bool(success),
                depth=context.depth,
            )

        return RLMResult(
            answer=str(response),
            success=True,
            depth=context.depth,
        )

