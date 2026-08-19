"""
Control-flow contract tests for NanoCodeAgent.

These replace the script-style checks with ISOLATED, DETERMINISTIC tests that
make NO live LLM calls. Each test builds a fresh agent with injected fakes, so
tests run in any order and one failure does not abort the rest.

Runs two ways:
    python test_agent_contract.py     # built-in runner: prints per-test result + summary, exits non-zero on failure
    pytest test_agent_contract.py     # if pytest is installed (the test_* functions are plain asserts)

Why this file exists
--------------------
RLM will slot between the planner and the executor, and RSI will mutate the
prompts/policies wrapped around this pipeline. Both need a regression net that
pins the *base* control flow so a change can be judged against a fixed
reference. The most important invariant here is the human-rejection path
(rejection is NOT a failure: no evaluate, no reflect, no retry, no memory
write) — which currently has zero coverage.

Scope note: these are contract tests for the ORCHESTRATION layer only. Planner,
executor, evaluator and reflector are replaced by fakes, so component-emitted
trace events (evaluation.completed, reflection.completed) are intentionally not
asserted here — they belong in each component's own unit test. Only
agent-emitted events (memory.*) are checked.
"""

from agent.agent import NanoCodeAgent
from agent.memory import Experience
from agent.state import AgentStatus
from models.config import AgentConfig


# ---------------------------------------------------------------------------
# Fakes (build on the dependency-injection pattern already used in your
# test_agent_retry.py / test_agent_memory_integration.py).
# ---------------------------------------------------------------------------

class RecordingPlanner:
    """Records what the agent hands the planner on each attempt."""

    def __init__(self):
        self.calls = []

    def run(self, state, experiences=None, retry_context=None):
        self.calls.append({"experiences": experiences, "retry_context": retry_context})


class ScriptedExecutor:
    """Sets final_response each call. Can force a status (e.g. rejection)."""

    def __init__(self, force_status=None):
        self.calls = 0
        self.force_status = force_status

    def run(self, state):
        self.calls += 1
        state.final_response = f"response-{self.calls}"
        if self.force_status is not None:
            state.status = self.force_status


class _Evaluation:
    def __init__(self, success):
        self.success = success
        self.score = 1.0 if success else 0.0
        self.reason = "Task completed successfully." if success else "Task failed."


class ScriptedEvaluator:
    """Returns a scripted success/failure sequence; last value repeats."""

    def __init__(self, successes):
        self.successes = successes
        self.calls = 0

    def evaluate(self, state):
        value = self.successes[min(self.calls, len(self.successes) - 1)]
        self.calls += 1
        return _Evaluation(value)


class _Reflection:
    def __init__(self, success):
        # Mirrors the real reflector: success -> no improvement needed.
        self.should_improve = not success
        self.diagnosis = "The task was completed successfully." if success else "The first attempt used the wrong approach."
        self.improvement = "No improvement required." if success else "Use a better approach on the next attempt."


class RecordingReflector:
    def __init__(self):
        self.calls = 0

    def reflect(self, state, evaluation):
        self.calls += 1
        return _Reflection(evaluation.success)


# ---------------------------------------------------------------------------
# Human-rejection mechanism — SINGLE KNOB.
#
# This file assumes the executor signals a human rejection by setting
# state.status = AgentStatus.HUMAN_REJECTED, and that the agent then stops
# before evaluation. If your build signals rejection differently (e.g. the
# approval layer raises an ApprovalRejected exception the agent catches),
# change ONLY this helper and the rejection tests keep working.
#
# If the rejection tests fail and this knob is already correct, that is a real
# finding: the base agent is falling through to the evaluator on rejection,
# i.e. treating a human "no" as an ordinary attempt.
# ---------------------------------------------------------------------------

REJECTION_STATUS = AgentStatus.HUMAN_REJECTED


def make_rejecting_executor():
    return ScriptedExecutor(force_status=REJECTION_STATUS)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_agent(*, successes=(True,), max_retries=2, executor=None, seed=()):
    agent = NanoCodeAgent(config=AgentConfig(max_retries=max_retries))
    agent.planner = RecordingPlanner()
    agent.executor = executor if executor is not None else ScriptedExecutor()
    agent.evaluator = ScriptedEvaluator(list(successes))
    agent.reflector = RecordingReflector()
    for exp in seed:
        agent.memory.add(exp)
    return agent


def _names(agent):
    return [e.name for e in agent.tracer.get_events()]


# ---------------------------------------------------------------------------
# Success / retry control flow
# ---------------------------------------------------------------------------

def test_success_runs_pipeline_once():
    agent = build_agent(successes=[True], max_retries=2)
    response = agent.run("What is Python?")

    assert response == "response-1"
    assert agent.executor.calls == 1
    assert agent.evaluator.calls == 1
    assert agent.reflector.calls == 1
    assert len(agent.planner.calls) == 1


def test_failure_triggers_single_retry():
    agent = build_agent(successes=[False, True], max_retries=2)
    response = agent.run("Fix the failing test")

    assert response == "response-2"
    assert agent.executor.calls == 2
    assert agent.evaluator.calls == 2
    assert agent.reflector.calls == 2
    assert len(agent.planner.calls) == 2


def test_retry_context_reaches_planner_on_second_attempt():
    agent = build_agent(successes=[False, True], max_retries=2)
    agent.run("Fix the failing test")

    first, second = agent.planner.calls[0], agent.planner.calls[1]
    assert first["retry_context"] is None
    assert second["retry_context"] is not None
    assert second["retry_context"]["diagnosis"] == "The first attempt used the wrong approach."
    assert second["retry_context"]["improvement"] == "Use a better approach on the next attempt."


def test_retry_stops_at_limit():
    agent = build_agent(successes=[False, False, False], max_retries=2)
    response = agent.run("Fix the failing test")

    # initial attempt + max_retries
    assert response == "response-3"
    assert agent.executor.calls == 3
    assert agent.evaluator.calls == 3
    assert len(agent.planner.calls) == 3


# ---------------------------------------------------------------------------
# Memory plumbing
# ---------------------------------------------------------------------------

def test_seeded_experience_reaches_planner_on_first_attempt():
    seed = [Experience(
        task="Fix a Python test",
        diagnosis="The test failed because the wrong file was used.",
        improvement="Inspect the project structure before running the test.",
        success=False,
    )]
    agent = build_agent(successes=[True], max_retries=1, seed=seed)
    agent.run("Fix another failing Python test")

    first = agent.planner.calls[0]
    assert first["experiences"]
    assert len(first["experiences"]) >= 1
    assert first["experiences"][0].task == "Fix a Python test"


def test_empty_memory_passes_empty_experiences():
    agent = build_agent(successes=[True], max_retries=1)
    agent.run("What is Python?")
    # empty list, not None (matches your integration test)
    assert agent.planner.calls[0]["experiences"] == []


def test_failed_run_stores_experience():
    agent = build_agent(successes=[False], max_retries=0)  # single attempt, fails
    before = agent.memory.count()
    agent.run("Fix the failing test")

    assert agent.memory.count() == before + 1
    stored = agent.memory.get_all()[-1]
    assert stored.success is False
    assert stored.diagnosis
    assert stored.improvement


def test_successful_run_stores_nothing():
    agent = build_agent(successes=[True], max_retries=0)
    agent.run("What is Python?")
    assert agent.memory.count() == 0


# ---------------------------------------------------------------------------
# Tracing (agent-emitted events only)
# ---------------------------------------------------------------------------

def test_memory_retrieved_is_traced():
    agent = build_agent(successes=[True], max_retries=0)
    agent.run("What is Python?")
    assert "memory.retrieved" in _names(agent)


def test_memory_stored_is_traced_on_failure():
    agent = build_agent(successes=[False], max_retries=0)
    agent.run("Fix the failing test")

    stored = [e for e in agent.tracer.get_events() if e.name == "memory.stored"]
    assert len(stored) >= 1
    assert stored[-1].data.get("success") is False


# ---------------------------------------------------------------------------
# Human-in-the-loop rejection — the load-bearing invariant, currently untested.
# Rejection must NOT be treated as a normal attempt.
# ---------------------------------------------------------------------------

# Rejection tests script a FAILURE evaluation on purpose: if the agent wrongly
# ignored the rejection and fell through, a failure would make it retry and
# store an experience — so these assertions genuinely bite instead of passing
# vacuously.

def test_human_rejection_skips_evaluation_and_reflection():
    agent = build_agent(successes=[False], max_retries=2, executor=make_rejecting_executor())
    agent.run("Delete the production database")

    assert agent.evaluator.calls == 0, "evaluator must not run after human rejection"
    assert agent.reflector.calls == 0, "reflector must not run after human rejection"


def test_human_rejection_does_not_retry():
    agent = build_agent(successes=[False], max_retries=3, executor=make_rejecting_executor())
    agent.run("Delete the production database")

    assert agent.executor.calls == 1, "human rejection must stop, not retry"


def test_human_rejection_writes_nothing_to_memory():
    agent = build_agent(successes=[False], max_retries=2, executor=make_rejecting_executor())
    agent.run("Delete the production database")

    assert agent.memory.count() == 0, "a rejected attempt is not an experience to learn from"


def test_human_rejection_returns_cleanly():
    agent = build_agent(successes=[False], max_retries=2, executor=make_rejecting_executor())
    response = agent.run("Delete the production database")
    # Should return control without raising; response is whatever was produced pre-rejection.
    assert isinstance(response, str)


# ---------------------------------------------------------------------------
# Built-in runner (works without pytest; mirrors your ✅ style)
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            print(f"❌ {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"💥 {test.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"✅ {test.__name__}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
