"""Tests for the recursive self-improvement (RSI) loop.

Deterministic: fake planner/executor/evaluator/reflector. No OpenRouter, no
LLM, no network.

Assertions are on recorded execution state -- the exact planner inputs, the
emitted trace events, the stored experiences -- not on "was this called".
"""

import ast

import pytest

from agent.agent import NanoCodeAgent
from agent.planner import Planner
from agent.rsi import RSIContext
from agent.state import AgentStatus
from agent.tracer import Tracer
from models.config import AgentConfig


TASK = "Fix the failing authentication test."


class RecordingPlanner:
    """Records the exact context each planning attempt received."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, state, experiences=None, retry_context=None):
        self.calls.append(
            {
                "task": state.task,
                "experiences": list(experiences or []),
                "retry_context": retry_context,
                # The real prompt the planner would build, so "did the input
                # actually change" can be asserted on real text.
                "prompt": Planner(tracer=Tracer(enabled=False))._build_user_prompt(
                    task=state.task,
                    experiences=list(experiences or []),
                    retry_context=retry_context,
                ),
            }
        )


class RecordingExecutor:
    def __init__(self, status=AgentStatus.COMPLETED) -> None:
        self.calls = 0
        self.status = status

    def run(self, state):
        self.calls += 1
        state.status = self.status
        state.final_response = f"response-{self.calls}"


class RejectingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, state):
        self.calls += 1
        state.status = AgentStatus.HUMAN_REJECTED
        state.final_response = "rejected by human"


class FakeEvaluation:
    def __init__(self, success: bool, reason: str = "") -> None:
        self.success = success
        self.score = 1.0 if success else 0.0
        self.reason = reason or (
            "Task completed successfully."
            if success
            else "Tests still fail because token expiry logic was not corrected."
        )


class ScriptedEvaluator:
    """Returns a scripted verdict per attempt."""

    def __init__(self, results: list[bool], reasons: list[str] | None = None) -> None:
        self.results = results
        self.reasons = reasons or []
        self.calls = 0

    def evaluate(self, state):
        index = self.calls
        self.calls += 1

        reason = self.reasons[index] if index < len(self.reasons) else ""

        return FakeEvaluation(self.results[index], reason)


class FakeReflection:
    def __init__(self, should_improve=True, diagnosis="", improvement="") -> None:
        self.should_improve = should_improve
        self.diagnosis = diagnosis
        self.improvement = improvement


class ScriptedReflector:
    """Produces a distinct reflection per attempt, so propagation is provable."""

    def __init__(self, reflections: list[FakeReflection] | None = None) -> None:
        self.reflections = reflections
        self.calls = 0

    def reflect(self, state, evaluation):
        index = self.calls
        self.calls += 1

        if self.reflections is not None:
            return self.reflections[min(index, len(self.reflections) - 1)]

        if evaluation.success:
            return FakeReflection(
                should_improve=False,
                diagnosis="The task was completed successfully.",
                improvement="No improvement required.",
            )

        return FakeReflection(
            should_improve=True,
            diagnosis=f"attempt {index + 1} modified token creation only",
            improvement=f"attempt {index + 1}: fix the timestamp-unit conversion",
        )


def build_agent(
    evaluations: list[bool],
    max_retries: int = 2,
    reflections=None,
    executor=None,
    reasons=None,
):
    """An agent whose pipeline stages are deterministic recorders."""

    events = []

    agent = NanoCodeAgent(
        config=AgentConfig(max_retries=max_retries),
        console_trace=False,
        trace_callback=events.append,
        rlm_enabled=False,
    )

    agent.planner = RecordingPlanner()
    agent.executor = executor if executor is not None else RecordingExecutor()
    agent.evaluator = ScriptedEvaluator(evaluations, reasons)
    agent.reflector = ScriptedReflector(reflections)

    return agent, events


def names(events) -> list[str]:
    return [event.name for event in events]


def event_named(events, name):
    return [event for event in events if event.name == name]


# ---------------------------------------------------------------------------
# 1. A successful first attempt does not trigger RSI
# ---------------------------------------------------------------------------

def test_successful_first_attempt_runs_once_without_rsi():

    agent, events = build_agent([True])

    response = agent.run(TASK)

    assert response == "response-1"

    assert len(agent.planner.calls) == 1
    assert agent.executor.calls == 1
    assert agent.evaluator.calls == 1

    # No retry, and no RSI activity at all.
    assert agent.planner.calls[0]["retry_context"] is None
    assert not any(name.startswith("rsi.") for name in names(events))
    assert not any(name.startswith("retry.") for name in names(events))

    assert agent.last_rsi_context.attempt == 1
    assert agent.last_rsi_context.is_retry is False


# ---------------------------------------------------------------------------
# 2-3. A failed attempt reflects, retries with improvement, and succeeds
# ---------------------------------------------------------------------------

def test_failed_attempt_retries_with_improvement_context():

    agent, events = build_agent([False, True])

    response = agent.run(TASK)

    assert response == "response-2"

    assert len(agent.planner.calls) == 2
    assert agent.reflector.calls == 2

    first, second = agent.planner.calls

    assert first["retry_context"] is None

    context = second["retry_context"]

    assert context is not None
    assert context["attempt"] == "2"
    assert context["diagnosis"] == "attempt 1 modified token creation only"
    assert context["improvement"] == (
        "attempt 1: fix the timestamp-unit conversion"
    )


def test_a_successful_retry_reports_success_and_the_attempt_count():

    agent, events = build_agent([False, True])

    agent.run(TASK)

    assert names(events).count("retry.started") == 1
    assert event_named(events, "retry.completed")[0].data["attempt"] == 1

    completed = event_named(events, "rsi.completed")

    assert len(completed) == 1
    assert completed[0].data["success"] is True
    assert completed[0].data["attempt"] == 2

    assert agent.last_rsi_context.attempt == 2
    assert agent.last_rsi_context.is_retry is True


# ---------------------------------------------------------------------------
# 4 & 11. Retry limits are respected exactly, with no extra attempt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_exactly_the_configured_number_of_retries_runs(max_retries):

    agent, events = build_agent([False] * (max_retries + 2), max_retries=max_retries)

    agent.run(TASK)

    expected_attempts = max_retries + 1

    assert len(agent.planner.calls) == expected_attempts
    assert agent.executor.calls == expected_attempts
    assert agent.evaluator.calls == expected_attempts

    assert names(events).count("retry.started") == max_retries
    assert names(events).count("rsi.retry.started") == max_retries

    assert len(event_named(events, "rsi.exhausted")) == 1


def test_rsi_does_not_keep_its_own_retry_counter():

    agent, events = build_agent([False, False, False, False], max_retries=2)

    agent.run(TASK)

    exhausted = event_named(events, "rsi.exhausted")[0]

    assert exhausted.data["max_retries"] == 2
    assert exhausted.data["attempts"] == 3

    # The RSI attempt number and the existing retry counter stay in lockstep.
    assert agent.last_rsi_context.attempt == 3
    assert agent.executor.calls == 3


# ---------------------------------------------------------------------------
# 5. Improvement context propagation, attempt by attempt
# ---------------------------------------------------------------------------

def test_each_attempt_receives_the_previous_attempts_reflection():

    agent, _ = build_agent([False, False, True], max_retries=2)

    agent.run(TASK)

    assert len(agent.planner.calls) == 3

    second = agent.planner.calls[1]["retry_context"]
    third = agent.planner.calls[2]["retry_context"]

    # Attempt 2 carries attempt 1's reflection, attempt 3 carries attempt 2's.
    assert second["diagnosis"] == "attempt 1 modified token creation only"
    assert third["diagnosis"] == "attempt 2 modified token creation only"

    assert second["attempt"] == "2"
    assert third["attempt"] == "3"


def test_the_evaluation_result_reaches_the_next_attempt():

    agent, _ = build_agent(
        [False, True],
        reasons=["Tests still fail: token expiry is computed in milliseconds."],
    )

    agent.run(TASK)

    context = agent.planner.calls[1]["retry_context"]

    assert context["evaluation"] == (
        "Tests still fail: token expiry is computed in milliseconds."
    )


def test_the_previous_response_reaches_the_next_attempt():

    agent, _ = build_agent([False, True])

    agent.run(TASK)

    assert agent.planner.calls[1]["retry_context"]["previous_response"] == (
        "response-1"
    )


def test_reflection_text_reaches_the_actual_planner_prompt():

    agent, _ = build_agent([False, True])

    agent.run(TASK)

    prompt = agent.planner.calls[1]["prompt"]

    assert "Previous attempt failed:" in prompt
    assert "attempt 1 modified token creation only" in prompt
    assert "attempt 1: fix the timestamp-unit conversion" in prompt
    assert "Attempt: 2" in prompt
    assert TASK in prompt


# ---------------------------------------------------------------------------
# 6. No blind repetition
# ---------------------------------------------------------------------------

def test_the_retry_prompt_differs_from_the_first_attempt():

    agent, _ = build_agent([False, True])

    agent.run(TASK)

    first = agent.planner.calls[0]["prompt"]
    second = agent.planner.calls[1]["prompt"]

    assert first != second

    # The first attempt is the bare task; the second carries what was learned.
    assert first == TASK
    assert "attempt 1 modified token creation only" in second


def test_every_attempt_has_a_distinct_prompt():

    agent, _ = build_agent([False, False, True], max_retries=2)

    agent.run(TASK)

    prompts = [call["prompt"] for call in agent.planner.calls]

    assert len(set(prompts)) == len(prompts)


def test_the_prompt_still_differs_when_reflection_is_empty():

    # Reflection wants a retry but has nothing to say.
    empty = FakeReflection(should_improve=True, diagnosis="", improvement="")

    agent, _ = build_agent([False, True], reflections=[empty, empty])

    agent.run(TASK)

    first, second = (call["prompt"] for call in agent.planner.calls)

    # Attempt number and evaluation still distinguish the attempts; no
    # guidance is invented to fill the gap.
    assert first != second
    assert "Attempt: 2" in second


# ---------------------------------------------------------------------------
# 7. Human rejection stops everything immediately
# ---------------------------------------------------------------------------

def test_human_rejection_stops_without_reflection_retry_or_rsi():

    agent, events = build_agent([True], executor=RejectingExecutor())

    response = agent.run(TASK)

    assert response == "rejected by human"

    # Rejection happens before evaluation, so nothing downstream ran.
    assert agent.executor.calls == 1
    assert agent.evaluator.calls == 0
    assert agent.reflector.calls == 0
    assert len(agent.planner.calls) == 1

    assert not any(name.startswith("rsi.") for name in names(events))
    assert not any(name.startswith("retry.") for name in names(events))

    assert "agent.stopped" in names(events)

    # And nothing was written to memory.
    assert agent.memory.experiences == []


# ---------------------------------------------------------------------------
# 8. Missing or unusable reflection preserves the existing retry behavior
# ---------------------------------------------------------------------------

def test_reflection_without_improvement_stops_the_loop():

    no_improve = FakeReflection(
        should_improve=False,
        diagnosis="Nothing to improve.",
        improvement="",
    )

    agent, events = build_agent([False, True], reflections=[no_improve])

    agent.run(TASK)

    # Existing behaviour: should_improve=False ends the loop after one attempt.
    assert agent.executor.calls == 1
    assert names(events).count("retry.started") == 0

    reflection_events = event_named(events, "rsi.reflection.completed")

    assert len(reflection_events) == 1
    assert reflection_events[0].data["has_improvement"] is False
    assert reflection_events[0].data["should_improve"] is False


def test_empty_reflection_guidance_is_reported_not_invented():

    empty = FakeReflection(should_improve=True, diagnosis="   ", improvement="")

    agent, events = build_agent([False, True], reflections=[empty, empty])

    agent.run(TASK)

    context = agent.planner.calls[1]["retry_context"]

    assert context["diagnosis"] == ""
    assert context["improvement"] == ""

    assert event_named(events, "rsi.started")[0].data["has_improvement"] is False


def test_rsi_context_handles_a_missing_reflection():

    context = RSIContext(attempt=2, previous_evaluation=None, previous_reflection=None)

    assert context.is_retry is True
    assert context.has_improvement is False

    planner_context = context.to_planner_context()

    assert planner_context["attempt"] == "2"
    assert planner_context["diagnosis"] == ""
    assert "evaluation" not in planner_context


def test_the_first_rsi_context_has_no_planner_context():

    context = RSIContext()

    assert context.attempt == 1
    assert context.is_retry is False
    assert context.to_planner_context() is None


# ---------------------------------------------------------------------------
# 9. Tracing
# ---------------------------------------------------------------------------

def test_rsi_lifecycle_events_are_emitted_in_order():

    agent, events = build_agent([False, False, True], max_retries=2)

    agent.run(TASK)

    rsi_names = [name for name in names(events) if name.startswith("rsi.")]

    assert rsi_names == [
        "rsi.reflection.completed",
        "rsi.started",
        "rsi.retry.started",
        "rsi.reflection.completed",
        "rsi.retry.started",
        "rsi.completed",
    ]


def test_rsi_events_carry_useful_metadata():

    agent, events = build_agent([False, True])

    agent.run(TASK)

    started = event_named(events, "rsi.started")[0]

    assert started.data["attempt"] == 1
    assert started.data["reason"] == "evaluation_failed"
    assert started.data["has_improvement"] is True
    assert started.component == "rsi"

    retry = event_named(events, "rsi.retry.started")[0]

    assert retry.data["attempt"] == 2
    assert retry.data["is_retry"] is True
    assert retry.data["has_improvement"] is True


def test_rsi_events_do_not_expose_prompt_text():

    agent, events = build_agent([False, True])

    agent.run(TASK)

    for event in events:
        if not event.name.startswith("rsi."):
            continue

        payload = " ".join(str(value) for value in event.data.values())

        # Summaries carry counts and flags, never the built prompt.
        assert "Previous attempt failed:" not in payload
        assert "Current task:" not in payload


def test_the_existing_retry_events_are_preserved():

    agent, events = build_agent([False, False, False], max_retries=2)

    agent.run(TASK)

    event_names = names(events)

    assert event_names.count("retry.started") == 2
    assert "retry.exhausted" in event_names


def calls_print(node) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "print"
        for child in ast.walk(node)
    )


def test_the_agent_pipeline_contains_no_print_calls():

    # Whole modules: RSI state and planning must be render-free.
    for module_path in ("agent/rsi.py", "agent/planner.py"):
        tree = ast.parse(open(module_path, encoding="utf-8").read())

        assert not calls_print(tree), module_path

    # agent.py also holds a legacy standalone REPL (print_banner / main) that
    # predates this phase, so the assertion is scoped to the agent class the
    # RSI loop lives in.
    tree = ast.parse(open("agent/agent.py", encoding="utf-8").read())

    agent_class = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NanoCodeAgent"
    ][0]

    assert not calls_print(agent_class)


def test_the_agent_layer_does_not_import_the_cli():

    source = open("agent/rsi.py", encoding="utf-8").read()

    tree = ast.parse(source)

    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(module.startswith("cli") for module in imported)


# ---------------------------------------------------------------------------
# 10. RLM compatibility
# ---------------------------------------------------------------------------

def test_rsi_does_not_re_enable_rlm_for_child_agents():

    from rlm.nanocode_handler import create_nanocode_agent

    child = create_nanocode_agent()

    assert child.rlm_enabled is False
    assert child.router is None

    # A child still has its own RSI state: self-improvement works inside a
    # child without spawning another RLM hierarchy.
    assert child.last_rsi_context is None


def test_rsi_runs_inside_a_child_agent_without_nesting_rlm():

    agent, events = build_agent([False, True])

    # rlm_enabled=False is what a child agent runs with.
    assert agent.rlm_enabled is False
    assert agent.router is None

    agent.run(TASK)

    assert agent.executor.calls == 2
    assert "rsi.completed" in names(events)

    # No RLM path was entered.
    assert agent.rlm_orchestrator is None
    assert agent.last_route_decision is None


# ---------------------------------------------------------------------------
# 12. Memory correctness
# ---------------------------------------------------------------------------

def test_a_failed_final_run_is_not_stored_as_a_success():

    agent, _ = build_agent([False, False, False], max_retries=2)

    agent.run(TASK)

    assert len(agent.memory.experiences) == 1

    experience = agent.memory.experiences[0]

    assert experience.success is False
    assert experience.task == TASK

    # The stored lesson is the final attempt's reflection.
    assert experience.diagnosis == "attempt 3 modified token creation only"


def test_an_improved_successful_run_stores_no_failure():

    agent, events = build_agent([False, True])

    agent.run(TASK)

    # Existing memory contract: a run that ends successfully stores nothing,
    # even if it needed a retry to get there.
    assert agent.memory.experiences == []
    assert "memory.stored" not in names(events)


def test_a_successful_first_attempt_stores_nothing():

    agent, events = build_agent([True])

    agent.run(TASK)

    assert agent.memory.experiences == []
    assert "memory.stored" not in names(events)


def test_stored_failure_is_retrievable_for_a_later_task():

    agent, _ = build_agent([False, False, False], max_retries=2)

    agent.run(TASK)

    retrieved = agent.memory.retrieve(TASK)

    assert len(retrieved) == 1
    assert retrieved[0].success is False
