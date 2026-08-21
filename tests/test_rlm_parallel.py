"""Tests for bounded parallel RLM child execution.

Deterministic: concurrency is proven with threading primitives (Barrier,
Event, Lock), never with sleep() or wall-clock timing. No OpenRouter, no LLM,
no network.
"""

import ast
import threading

import pytest

from agent.tracer import Tracer
from cli.renderer import TerminalRenderer
from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.decomposer import DeterministicRLMDecomposer
from rlm.nanocode_handler import NanoCodeCallHandler
from rlm.orchestrator import RLMOrchestrator
from rlm.result import RLMResult
from rlm.runtime import DEFAULT_MAX_CONCURRENCY, RLMRuntime
from rlm.synthesizer import (
    FAILED_KEY,
    PARTIAL_KEY,
    RATE_LIMITED_KEY,
    RLMSynthesizer,
    SUCCESSFUL_KEY,
)


COMPLEX_TASK = "Find the authentication bug across the project."

# Generous enough that a genuinely sequential runtime fails the test by
# timing out rather than hanging the suite forever.
BARRIER_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Instrumented handlers
# ---------------------------------------------------------------------------

class ConcurrencyProbe(NanoCodeCallHandler):
    """Records how many children were inside `call` at the same time."""

    def __init__(self, agent_factory=None, barrier: threading.Barrier | None = None):
        super().__init__(
            agent_factory=agent_factory or (lambda: None),
            rate_limit_delay=0.0,
        )

        self.barrier = barrier

        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.order: list[str] = []
        self.completed: list[str] = []

    def call(self, context: RLMContext) -> RLMResult:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(context.task)

        try:
            if self.barrier is not None:
                # Every participant must arrive before any may leave: this
                # cannot pass unless the children genuinely overlap.
                self.barrier.wait(timeout=BARRIER_TIMEOUT)

            return RLMResult(
                answer=f"answer for {context.task}",
                success=True,
                depth=context.depth,
                metadata={"task": context.task},
            )
        finally:
            with self.lock:
                self.active -= 1
                self.completed.append(context.task)


class ReverseOrderHandler(NanoCodeCallHandler):
    """Forces children to finish in reverse order using explicit gates."""

    def __init__(self, child_count: int):
        super().__init__(agent_factory=lambda: None, rate_limit_delay=0.0)

        self.child_count = child_count
        self.gates = [threading.Event() for _ in range(child_count)]
        self.finished: list[str] = []
        self.lock = threading.Lock()
        self.index_of: dict[str, int] = {}
        self.assign_lock = threading.Lock()

    def _slot(self, task: str) -> int:
        with self.assign_lock:
            if task not in self.index_of:
                self.index_of[task] = len(self.index_of)

            return self.index_of[task]

    def call(self, context: RLMContext) -> RLMResult:
        position = self._slot(context.task)

        # The last child finishes first and releases the one before it.
        if position == self.child_count - 1:
            pass
        else:
            self.gates[position].wait(timeout=BARRIER_TIMEOUT)

        with self.lock:
            self.finished.append(context.task)

        if position > 0:
            self.gates[position - 1].set()

        return RLMResult(
            answer=f"answer for {context.task}",
            success=True,
            depth=context.depth,
            metadata={"task": context.task},
        )


def tasks_for(count: int) -> list[tuple[str, str]]:
    return [(f"child task {index}", COMPLEX_TASK) for index in range(1, count + 1)]


def build_runtime(handler, max_concurrency=DEFAULT_MAX_CONCURRENCY, budget=None):
    return RLMRuntime(
        call_handler=handler,
        budget=budget or RLMBudget(max_depth=2, max_children=6, max_iterations=12),
        max_concurrency=max_concurrency,
    )


# ---------------------------------------------------------------------------
# 1-5. Concurrency mechanism and its bound
# ---------------------------------------------------------------------------

def test_max_concurrency_one_reproduces_sequential_behavior():

    handler = ConcurrencyProbe()

    runtime = build_runtime(handler, max_concurrency=1)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert len(results) == 3

    # Never more than one child inside the handler at a time.
    assert handler.peak == 1

    # And they ran in order.
    assert handler.order == [task for task, _ in tasks_for(3)]


@pytest.mark.parametrize("limit", [2, 3])
def test_children_overlap_up_to_the_configured_limit(limit):

    # A barrier of `limit` participants only releases when that many children
    # are inside the handler simultaneously.
    barrier = threading.Barrier(limit)

    handler = ConcurrencyProbe(barrier=barrier)

    runtime = build_runtime(handler, max_concurrency=limit)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(limit))

    assert len(results) == limit
    assert all(result.success for result in results)

    assert handler.peak == limit
    assert runtime.peak_active_children == limit


def test_concurrency_never_exceeds_the_limit():

    handler = ConcurrencyProbe(barrier=threading.Barrier(2))

    runtime = build_runtime(handler, max_concurrency=2)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(6))

    assert handler.peak == 2
    assert runtime.peak_active_children == 2


def test_a_sequential_runtime_cannot_satisfy_a_two_party_barrier():

    # The negative control: with max_concurrency=1 the barrier can never be
    # satisfied, which is what makes the overlap tests meaningful.
    barrier = threading.Barrier(2)

    handler = ConcurrencyProbe(barrier=barrier)

    runtime = build_runtime(handler, max_concurrency=1)

    with pytest.raises(threading.BrokenBarrierError):
        runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(2))


@pytest.mark.parametrize("limit", [0, -1, -5])
def test_a_non_positive_concurrency_limit_is_rejected(limit):

    with pytest.raises(ValueError, match="max_concurrency"):
        RLMRuntime(call_handler=ConcurrencyProbe(), max_concurrency=limit)


def test_concurrency_does_not_depend_on_the_number_of_children():

    handler = ConcurrencyProbe(barrier=threading.Barrier(2))

    runtime = build_runtime(handler, max_concurrency=2)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(5))

    # Five children, still only two at a time.
    assert handler.peak == 2


def test_the_default_concurrency_is_three():

    assert DEFAULT_MAX_CONCURRENCY == 3

    runtime = RLMRuntime(call_handler=ConcurrencyProbe())

    assert runtime.max_concurrency == 3


# ---------------------------------------------------------------------------
# 6-7 & 27. Result ordering
# ---------------------------------------------------------------------------

def test_results_come_back_in_decomposition_order():

    handler = ConcurrencyProbe(barrier=threading.Barrier(3))

    runtime = build_runtime(handler, max_concurrency=3)

    tasks = tasks_for(3)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks)

    assert [result.metadata["task"] for result in results] == [
        task for task, _ in tasks
    ]


def test_reverse_completion_order_still_synthesizes_in_order():

    handler = ReverseOrderHandler(child_count=3)

    runtime = build_runtime(handler, max_concurrency=3)

    tasks = tasks_for(3)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks)

    # Children genuinely finished last-to-first...
    assert handler.finished == [task for task, _ in reversed(tasks)]

    # ...and synthesis still sees decomposition order.
    assert [result.metadata["task"] for result in results] == [
        task for task, _ in tasks
    ]


def test_synthesis_receives_ordered_results_after_all_workers_finish():

    class RecordingSynthesizer(RLMSynthesizer):
        def __init__(self):
            self.batches = []

        def synthesize(self, results):
            self.batches.append(list(results))
            return super().synthesize(results)

    handler = ReverseOrderHandler(child_count=3)
    synthesizer = RecordingSynthesizer()

    runtime = build_runtime(handler, max_concurrency=3)

    tasks = tasks_for(3)

    runtime.call_and_synthesize(
        parent=RLMContext(task=COMPLEX_TASK),
        tasks=tasks,
        synthesizer=synthesizer,
    )

    assert len(synthesizer.batches) == 1

    batch = synthesizer.batches[0]

    assert len(batch) == 3
    assert [result.metadata["task"] for result in batch] == [
        task for task, _ in tasks
    ]


def test_ordering_is_deterministic_across_runs():

    def run():
        handler = ConcurrencyProbe(barrier=threading.Barrier(3))
        runtime = build_runtime(handler, max_concurrency=3)

        results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

        return [result.answer for result in results]

    assert run() == run() == run()


# ---------------------------------------------------------------------------
# 8 & 23. Child identity
# ---------------------------------------------------------------------------

def build_identity_pipeline(agent_factory, max_concurrency=3):
    """An orchestrator whose child events land in a recording tracer."""

    events: list = []

    tracer = Tracer(enabled=True, console=False, on_event=events.append)

    handler = NanoCodeCallHandler(agent_factory=agent_factory, rate_limit_delay=0.0)

    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=handler,
            budget=RLMBudget(max_depth=2, max_children=6, max_iterations=12),
            max_concurrency=max_concurrency,
        ),
        tracer=tracer,
    )

    return orchestrator, events


class TracingChild:
    """A child agent that emits real events and finishes in reverse order."""

    order: list[str] = []
    gates: dict[int, threading.Event] = {}
    lock = threading.Lock()
    total = 3

    def __init__(self) -> None:
        self.tracer = Tracer(enabled=True, console=False, on_event=None)

    def run(self, task: str) -> str:
        with TracingChild.lock:
            position = len(TracingChild.order)
            TracingChild.order.append(task)

        self.tracer.record("planner.started", component="planner", task=task)

        # Later children finish first.
        if position < TracingChild.total - 1:
            TracingChild.gates[position].wait(timeout=BARRIER_TIMEOUT)

        self.tracer.record("tool.started", component="executor", tool="grep")
        self.tracer.record("executor.completed", component="agent", task=task)

        if position > 0:
            TracingChild.gates[position - 1].set()

        return f"auth.py finding from {task}"


def reset_tracing_child(total: int = 3) -> None:
    TracingChild.order = []
    TracingChild.gates = {index: threading.Event() for index in range(total)}
    TracingChild.total = total


def test_child_identity_follows_decomposition_order_not_completion_order():

    reset_tracing_child()

    orchestrator, events = build_identity_pipeline(lambda: TracingChild())

    orchestrator.run(COMPLEX_TASK)

    child_tasks = [child.task for child in orchestrator.last_child_tasks]

    started = [
        event for event in events if event.name == "rlm.child.started"
    ]

    assert len(started) == len(child_tasks)

    # Child N is always the Nth decomposed task.
    for event in started:
        index = event.data["child"]

        assert event.data["task"] == child_tasks[index - 1]
        assert event.data["of"] == len(child_tasks)


def test_forwarded_child_events_keep_their_own_identity():

    reset_tracing_child()

    orchestrator, events = build_identity_pipeline(lambda: TracingChild())

    orchestrator.run(COMPLEX_TASK)

    child_tasks = [child.task for child in orchestrator.last_child_tasks]

    forwarded = [event for event in events if event.data.get("rlm_child")]

    assert forwarded

    for event in forwarded:
        index = event.data["rlm_child"]

        assert event.data["rlm_child_task"] == child_tasks[index - 1]
        assert event.data["rlm_child_count"] == len(child_tasks)
        assert event.data["depth"] == 1

    # Every child is represented.
    assert {event.data["rlm_child"] for event in forwarded} == {1, 2, 3}


# ---------------------------------------------------------------------------
# 9-11. Failure isolation under concurrency
# ---------------------------------------------------------------------------

class SelectiveFailureHandler(NanoCodeCallHandler):
    """Fails one child while its siblings run concurrently."""

    def __init__(self, failing_task: str, barrier: threading.Barrier):
        super().__init__(agent_factory=lambda: None, rate_limit_delay=0.0)

        self.failing_task = failing_task
        self.barrier = barrier
        self.seen: list[str] = []
        self.lock = threading.Lock()

    def call(self, context: RLMContext) -> RLMResult:
        with self.lock:
            self.seen.append(context.task)

        self.barrier.wait(timeout=BARRIER_TIMEOUT)

        if context.task == self.failing_task:
            raise RuntimeError("child exploded")

        return RLMResult(
            answer=f"answer for {context.task}",
            success=True,
            depth=context.depth,
            metadata={"task": context.task},
        )


def test_a_failing_child_does_not_cancel_its_siblings():

    tasks = tasks_for(3)

    handler = SelectiveFailureHandler(
        failing_task=tasks[1][0],
        barrier=threading.Barrier(3),
    )

    runtime = build_runtime(handler, max_concurrency=3)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks)

    assert len(results) == 3
    assert sorted(handler.seen) == sorted(task for task, _ in tasks)

    assert [result.success for result in results] == [True, False, True]

    # The exception became a failed result in the failing child's own slot.
    assert results[1].answer == ""
    assert "child exploded" in results[1].metadata["error"]


def test_worker_exceptions_become_failed_results():

    class ExplodingHandler(NanoCodeCallHandler):
        def __init__(self):
            super().__init__(agent_factory=lambda: None)

        def call(self, context):
            raise RuntimeError("boom")

    runtime = build_runtime(ExplodingHandler(), max_concurrency=3)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert len(results) == 3
    assert all(result.success is False for result in results)
    assert all("boom" in result.metadata["error"] for result in results)


def test_partial_success_preserves_partial_metadata():

    tasks = tasks_for(3)

    handler = SelectiveFailureHandler(
        failing_task=tasks[1][0],
        barrier=threading.Barrier(3),
    )

    runtime = build_runtime(handler, max_concurrency=3)

    result = runtime.call_and_synthesize(
        parent=RLMContext(task=COMPLEX_TASK),
        tasks=tasks,
        synthesizer=RLMSynthesizer(),
    )

    assert result.success is True
    assert result.metadata[PARTIAL_KEY] is True
    assert result.metadata[SUCCESSFUL_KEY] == 2
    assert result.metadata[FAILED_KEY] == 1


def test_all_children_failing_preserves_the_existing_failure_shape():

    class AlwaysFailingHandler(NanoCodeCallHandler):
        def __init__(self):
            super().__init__(agent_factory=lambda: None)

        def call(self, context):
            return RLMResult(answer="", success=False, depth=context.depth)

    runtime = build_runtime(AlwaysFailingHandler(), max_concurrency=3)

    result = runtime.call_and_synthesize(
        parent=RLMContext(task=COMPLEX_TASK),
        tasks=tasks_for(3),
        synthesizer=RLMSynthesizer(),
    )

    assert result.success is False
    assert result.answer == ""
    assert result.children_created == 3


# ---------------------------------------------------------------------------
# 12-14. Rate limits under concurrency
# ---------------------------------------------------------------------------

def rate_limit_error():
    import httpx
    import openai

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    response = httpx.Response(429, request=request, json={"error": {"code": 429}})

    return openai.RateLimitError(
        "Error code: 429 - rate-limited upstream", response=response, body=None
    )


class RateLimitedAgent:
    """Raises 429 for its first `failures` attempts, per instance."""

    def __init__(self, failures: int):
        self.failures = failures
        self.attempts = 0
        self.lock = threading.Lock()

    def run(self, task: str) -> str:
        with self.lock:
            self.attempts += 1
            attempt = self.attempts

        if attempt <= self.failures:
            raise rate_limit_error()

        return f"answer for {task}"


def test_a_rate_limited_child_keeps_its_rate_limit_metadata():

    class AlwaysRateLimited:
        def run(self, task):
            raise rate_limit_error()

    handler = NanoCodeCallHandler(
        agent_factory=lambda: AlwaysRateLimited(),
        rate_limit_retries=1,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    runtime = build_runtime(handler, max_concurrency=3)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    for result in results:
        assert result.success is False
        assert result.metadata["error_type"] == "rate_limit"
        assert result.metadata["rate_limited"] is True
        assert result.metadata["attempts"] == 2


def test_rate_limit_retries_stay_bounded_per_child():

    attempts_by_task: dict[str, int] = {}
    lock = threading.Lock()

    class CountingAgent:
        def run(self, task):
            with lock:
                attempts_by_task[task] = attempts_by_task.get(task, 0) + 1

            raise rate_limit_error()

    handler = NanoCodeCallHandler(
        agent_factory=lambda: CountingAgent(),
        rate_limit_retries=2,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    runtime = build_runtime(handler, max_concurrency=3)

    tasks = tasks_for(3)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks)

    # Each child retried its own budget, and no child retried on another's
    # behalf: 3 attempts each, never 9 for one.
    assert attempts_by_task == {task: 3 for task, _ in tasks}


def test_retries_do_not_consume_extra_child_budget():

    handler = NanoCodeCallHandler(
        agent_factory=lambda: RateLimitedAgent(failures=1),
        rate_limit_retries=1,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    budget = RLMBudget(max_depth=2, max_children=3, max_iterations=6)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert len(results) == 3
    assert budget.children_created == 3
    assert budget.iterations == 3


def test_a_retrying_child_does_not_exceed_the_concurrency_limit():

    peak = {"value": 0, "active": 0}
    lock = threading.Lock()

    class RetryingAgent:
        def __init__(self):
            self.attempts = 0

        def run(self, task):
            with lock:
                peak["active"] += 1
                peak["value"] = max(peak["value"], peak["active"])

            try:
                self.attempts += 1

                if self.attempts == 1:
                    raise rate_limit_error()

                return f"answer for {task}"
            finally:
                with lock:
                    peak["active"] -= 1

    handler = NanoCodeCallHandler(
        agent_factory=lambda: RetryingAgent(),
        rate_limit_retries=1,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    runtime = build_runtime(handler, max_concurrency=2)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(4))

    # A retry reuses its child's slot rather than opening a new one.
    assert peak["value"] <= 2


# ---------------------------------------------------------------------------
# 15-19. Budgets, depth and empty decomposition
# ---------------------------------------------------------------------------

def test_max_children_limits_the_children_that_actually_start():

    handler = ConcurrencyProbe()

    budget = RLMBudget(max_depth=2, max_children=2, max_iterations=10)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    with pytest.raises(RuntimeError, match="budget"):
        runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    # Exactly two children ran; the third never started.
    assert budget.children_created == 2
    assert len(handler.order) == 2


def test_planned_children_are_not_counted_as_executed():

    handler = ConcurrencyProbe()

    budget = RLMBudget(max_depth=2, max_children=1, max_iterations=10)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    with pytest.raises(RuntimeError, match="budget"):
        runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert budget.children_created == 1
    assert len(handler.completed) == 1


def test_max_iterations_is_still_enforced():

    handler = ConcurrencyProbe()

    budget = RLMBudget(max_depth=2, max_children=6, max_iterations=2)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    with pytest.raises(RuntimeError, match="budget"):
        runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(4))

    assert budget.iterations == 2


def test_max_depth_blocks_children_before_a_worker_starts():

    handler = ConcurrencyProbe()

    budget = RLMBudget(max_depth=2, max_children=6, max_iterations=12)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    parent = RLMContext(task=COMPLEX_TASK, depth=2)

    with pytest.raises(RuntimeError, match="budget"):
        runtime.call_many(parent, tasks_for(3))

    # No worker ever ran, and no budget was consumed.
    assert handler.order == []
    assert budget.children_created == 0


def test_empty_decomposition_starts_no_workers():

    handler = ConcurrencyProbe()

    budget = RLMBudget(max_depth=2, max_children=6, max_iterations=12)

    runtime = build_runtime(handler, max_concurrency=3, budget=budget)

    assert runtime.call_many(RLMContext(task=COMPLEX_TASK), []) == []

    assert handler.order == []
    assert budget.children_created == 0
    assert budget.iterations == 0


def test_sibling_contexts_stay_isolated():

    contexts: list[RLMContext] = []
    lock = threading.Lock()

    class MutatingHandler(NanoCodeCallHandler):
        def __init__(self):
            super().__init__(agent_factory=lambda: None)
            self.barrier = threading.Barrier(3)

        def call(self, context):
            with lock:
                contexts.append(context)

            self.barrier.wait(timeout=BARRIER_TIMEOUT)

            # Each child scribbles on its own metadata.
            context.metadata[f"touched_by_{context.task}"] = True

            return RLMResult(answer="ok", success=True, depth=context.depth)

    parent = RLMContext(task=COMPLEX_TASK, metadata={"run": "abc"})

    runtime = build_runtime(MutatingHandler(), max_concurrency=3)

    runtime.call_many(parent, tasks_for(3))

    assert parent.metadata == {"run": "abc"}

    for context in contexts:
        own = f"touched_by_{context.task}"

        assert own in context.metadata

        # No sibling's mark leaked in.
        for other in contexts:
            if other is not context:
                assert f"touched_by_{other.task}" not in context.metadata


# ---------------------------------------------------------------------------
# 20-21. RSI inside parallel children
# ---------------------------------------------------------------------------

def test_rsi_attempts_stay_sequential_inside_a_child():

    overlap = {"active": 0, "peak": 0}
    lock = threading.Lock()

    # The handler builds a fresh agent per attempt, so attempts are counted
    # per task rather than per instance.
    attempts: dict[str, int] = {}

    class RSIChild:
        """A child whose two attempts must never overlap with each other."""

        def run(self, task):
            with lock:
                overlap["active"] += 1
                overlap["peak"] = max(overlap["peak"], overlap["active"])
                attempts[task] = attempts.get(task, 0) + 1
                attempt = attempts[task]

            try:
                if attempt == 1:
                    raise rate_limit_error()

                return f"answer for {task}"
            finally:
                with lock:
                    overlap["active"] -= 1

    handler = NanoCodeCallHandler(
        agent_factory=lambda: RSIChild(),
        rate_limit_retries=1,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    runtime = build_runtime(handler, max_concurrency=1)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(2))

    assert all(result.success for result in results)

    # Each child needed a second attempt...
    assert attempts == {task: 2 for task, _ in tasks_for(2)}

    # ...and with one execution slot, no two attempts ever ran at once.
    assert overlap["peak"] == 1


def test_a_retrying_child_runs_alongside_other_children():

    started = threading.Barrier(2)
    seen: list[str] = []
    lock = threading.Lock()

    attempts: dict[str, int] = {}

    class SlowRetryAgent:
        def run(self, task):
            with lock:
                attempts[task] = attempts.get(task, 0) + 1
                attempt = attempts[task]
                seen.append(f"{task}:{attempt}")

            if task.endswith("1") and attempt == 1:
                # Wait for the sibling to start before failing, proving they
                # overlap rather than queueing.
                started.wait(timeout=BARRIER_TIMEOUT)
                raise rate_limit_error()

            if task.endswith("2") and attempt == 1:
                started.wait(timeout=BARRIER_TIMEOUT)

            return f"answer for {task}"

    handler = NanoCodeCallHandler(
        agent_factory=lambda: SlowRetryAgent(),
        rate_limit_retries=1,
        rate_limit_delay=0.0,
        sleep=lambda delay: None,
    )

    runtime = build_runtime(handler, max_concurrency=2)

    results = runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(2))

    assert all(result.success for result in results)

    # Child 1 retried while child 2 was already running.
    assert any(entry.endswith(":2") for entry in seen)


# ---------------------------------------------------------------------------
# 22 & 24. Tracing and rendering under interleaving
# ---------------------------------------------------------------------------

def test_the_parent_tracer_records_interleaved_events_safely():

    tracer = Tracer(enabled=True, console=False, on_event=None)

    barrier = threading.Barrier(4)

    def emit(child: int) -> None:
        barrier.wait(timeout=BARRIER_TIMEOUT)

        for index in range(50):
            tracer.record(
                "tool.started",
                component="executor",
                tool=f"tool-{index}",
                rlm_child=child,
            )

    threads = [threading.Thread(target=emit, args=(child,)) for child in range(1, 5)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT)

    # Nothing was lost or corrupted by concurrent delivery.
    assert len(tracer.events) == 200

    for child in range(1, 5):
        assert sum(
            1 for event in tracer.events if event.data["rlm_child"] == child
        ) == 50


def test_callbacks_are_delivered_one_at_a_time():

    concurrent_callbacks = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def callback(event) -> None:
        with lock:
            concurrent_callbacks["active"] += 1
            concurrent_callbacks["peak"] = max(
                concurrent_callbacks["peak"], concurrent_callbacks["active"]
            )

        with lock:
            concurrent_callbacks["active"] -= 1

    tracer = Tracer(enabled=True, console=False, on_event=callback)

    barrier = threading.Barrier(3)

    def emit() -> None:
        barrier.wait(timeout=BARRIER_TIMEOUT)

        for _ in range(100):
            tracer.record("tool.started", component="executor", tool="grep")

    threads = [threading.Thread(target=emit) for _ in range(3)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT)

    assert concurrent_callbacks["peak"] == 1
    assert len(tracer.events) == 300


def test_the_renderer_attributes_interleaved_events_to_the_right_child(capsys):

    renderer = TerminalRenderer()

    tracer = Tracer(enabled=True, console=False, on_event=renderer.handle_trace)

    def child_event(name: str, child: int, **data):
        tracer.record(
            name,
            component="executor",
            rlm_child=child,
            rlm_child_count=3,
            rlm_child_task=f"child task {child}",
            **data,
        )

    # Deliberately interleaved, as parallel children produce.
    child_event("planner.started", 1)
    child_event("planner.started", 2)
    child_event("tool.started", 2, tool="read_file")
    child_event("tool.started", 1, tool="grep")
    child_event("executor.completed", 2)
    child_event("executor.completed", 1)

    output = capsys.readouterr().out

    assert "[RLM child 1/3] Planning..." in output
    assert "[RLM child 2/3] Planning..." in output
    assert "[RLM child 2/3] read_file" in output
    assert "[RLM child 1/3] grep" in output

    # Child 2's tool is never attributed to child 1.
    assert "[RLM child 1/3] read_file" not in output
    assert "[RLM child 2/3] grep" not in output


def test_the_renderer_survives_concurrent_delivery(capsys):

    renderer = TerminalRenderer()

    tracer = Tracer(enabled=True, console=False, on_event=renderer.handle_trace)

    barrier = threading.Barrier(3)

    def emit(child: int) -> None:
        barrier.wait(timeout=BARRIER_TIMEOUT)

        for _ in range(20):
            tracer.record(
                "tool.started",
                component="executor",
                tool=f"tool{child}",
                rlm_child=child,
                rlm_child_count=3,
            )

    threads = [threading.Thread(target=emit, args=(child,)) for child in (1, 2, 3)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT)

    output = capsys.readouterr().out

    for child in (1, 2, 3):
        # Each line pairs the right child with the right tool.
        assert f"[RLM child {child}/3] tool{child}" in output

        for other in (1, 2, 3):
            if other != child:
                assert f"[RLM child {child}/3] tool{other}" not in output


# ---------------------------------------------------------------------------
# 25 & 29-30. No stdout coupling, no thread leaks
# ---------------------------------------------------------------------------

def test_the_rlm_layer_still_has_no_stdout_coupling():

    for module_path in ("rlm/runtime.py", "rlm/orchestrator.py", "rlm/nanocode_handler.py"):
        source = open(module_path, encoding="utf-8").read()

        tree = ast.parse(source)

        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]

        assert "print" not in calls, module_path

        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }

        assert not any(module.startswith("cli") for module in imported)


def test_no_worker_threads_survive_a_run():

    before = threading.active_count()

    handler = ConcurrencyProbe(barrier=threading.Barrier(3))

    runtime = build_runtime(handler, max_concurrency=3)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    # The pool is closed on exit, so the worker threads are gone.
    assert threading.active_count() == before

    assert not any(
        thread.name.startswith("rlm-child")
        for thread in threading.enumerate()
    )


def test_workers_are_cleaned_up_even_when_children_fail():

    before = threading.active_count()

    class ExplodingHandler(NanoCodeCallHandler):
        def __init__(self):
            super().__init__(agent_factory=lambda: None)

        def call(self, context):
            raise RuntimeError("boom")

    runtime = build_runtime(ExplodingHandler(), max_concurrency=3)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert threading.active_count() == before


def test_the_active_child_counter_returns_to_zero():

    handler = ConcurrencyProbe(barrier=threading.Barrier(3))

    runtime = build_runtime(handler, max_concurrency=3)

    runtime.call_many(RLMContext(task=COMPLEX_TASK), tasks_for(3))

    assert runtime._active_children == 0
    assert runtime.peak_active_children == 3


# ---------------------------------------------------------------------------
# End-to-end through the orchestrator
# ---------------------------------------------------------------------------

def test_the_orchestrator_runs_children_in_parallel_end_to_end():

    barrier = threading.Barrier(3)

    class BarrierChild:
        def run(self, task: str) -> str:
            barrier.wait(timeout=BARRIER_TIMEOUT)
            return f"auth.py validate_token() finding for {task}"

    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=NanoCodeCallHandler(agent_factory=lambda: BarrierChild()),
            budget=RLMBudget(max_depth=2, max_children=4, max_iterations=8),
            max_concurrency=3,
        ),
    )

    answer = orchestrator.run(COMPLEX_TASK)

    assert answer
    assert orchestrator.last_result.success is True
    assert orchestrator.runtime.peak_active_children == 3


def test_the_default_orchestrator_runtime_is_parallel():

    orchestrator = RLMOrchestrator(agent_factory=lambda: None)

    runtime = orchestrator._create_runtime(child_count=3)

    assert runtime.max_concurrency == DEFAULT_MAX_CONCURRENCY


def test_a_sequential_orchestrator_can_be_configured():

    orchestrator = RLMOrchestrator(agent_factory=lambda: None, max_concurrency=1)

    assert orchestrator._create_runtime(child_count=3).max_concurrency == 1
