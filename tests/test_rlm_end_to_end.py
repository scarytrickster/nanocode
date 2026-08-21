"""End-to-end validation of the complete RLM pipeline.

This file proves the whole path works as one system:

    router -> orchestrator -> decomposer -> runtime -> handler
           -> child agents -> synthesizer -> final result

Everything is deterministic. No OpenRouter, no LLM, no network, no Langfuse
connection. The `no_llm_calls` fixture below is autouse, so any test in this
file that reached for a real model would fail loudly instead of silently
spending quota.

The tests here assert on *execution records* — what the handler was actually
called with — not on decomposer output, which is already covered by
tests/test_rlm_decomposition.py.
"""

import pytest

import config.settings as settings
from agent.agent import NanoCodeAgent
from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.decomposer import DeterministicRLMDecomposer, RLMDecomposer
from rlm.nanocode_handler import NanoCodeCallHandler
from rlm.orchestrator import RLMOrchestrator
from rlm.result import RLMResult
from rlm.router import STRATEGY_NORMAL, STRATEGY_RLM, RLMRouter
from rlm.runtime import RLMRuntime
from rlm.synthesizer import RLMSynthesizer


COMPLEX_TASK = "Find the authentication bug across the project."
SIMPLE_TASK = "What is Python?"

# Deterministic child answers, used to prove that the final answer is built
# from what the children actually returned.
CHILD_ANSWERS = [
    "auth.py uses milliseconds.",
    "validation compares against seconds.",
    "The expiry calculation therefore becomes negative.",
]


# ---------------------------------------------------------------------------
# Test 14 — no real LLM, enforced for every test in this module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_llm_calls(monkeypatch):
    """Any OpenRouter call from these tests is a test-architecture failure."""

    def explode(*args, **kwargs):
        raise AssertionError("Phase 6.8 tests must not call a real LLM")

    monkeypatch.setattr(settings.client.chat.completions, "create", explode)


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------

class ScriptedChildAgent:
    """A child NanoCode stand-in returning canned answers in call order."""

    def __init__(self, answers: list[str], log: list[str]) -> None:
        self.answers = answers
        self.log = log

    def run(self, task: str) -> str:
        index = len(self.log)

        self.log.append(task)

        if index < len(self.answers):
            return self.answers[index]

        return f"child result for {task}"


class SelectiveFailureAgent:
    """A child that crashes only when its task contains a marker."""

    def __init__(self, fail_marker: str, log: list[str]) -> None:
        self.fail_marker = fail_marker
        self.log = log

    def run(self, task: str) -> str:
        self.log.append(task)

        if self.fail_marker in task:
            raise RuntimeError("child exploded")

        return f"child result for {task}"


class RecordingCallHandler(NanoCodeCallHandler):
    """The real handler, plus a record of every context it was called with."""

    def __init__(self, agent_factory) -> None:
        super().__init__(agent_factory=agent_factory)

        self.contexts: list[RLMContext] = []
        self.results: list[RLMResult] = []

    def call(self, context: RLMContext) -> RLMResult:
        self.contexts.append(context)

        result = super().call(context)

        self.results.append(result)

        return result


class RecordingSynthesizer(RLMSynthesizer):
    """The real synthesizer, plus a record of the batches it received."""

    def __init__(self) -> None:
        self.batches: list[list[RLMResult]] = []

    def synthesize(self, results):
        self.batches.append(list(results))

        return super().synthesize(results)


class EmptyDecomposer(RLMDecomposer):
    """A decomposer that produces nothing."""

    def decompose(self, task, context=None):
        return []


def build_pipeline(
    answers: list[str] | None = None,
    agent=None,
    budget: RLMBudget | None = None,
):
    """A complete RLM pipeline made of real components and fake children.

    Returns (orchestrator, handler, synthesizer, child_log).
    """

    child_log: list[str] = []

    child = agent if agent is not None else ScriptedChildAgent(
        answers if answers is not None else CHILD_ANSWERS,
        child_log,
    )

    handler = RecordingCallHandler(agent_factory=lambda: child)
    synthesizer = RecordingSynthesizer()

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget or RLMBudget(max_depth=2, max_children=5, max_iterations=10),
    )

    orchestrator = RLMOrchestrator(
        runtime=runtime,
        synthesizer=synthesizer,
    )

    return orchestrator, handler, synthesizer, getattr(child, "log", child_log)


# ---------------------------------------------------------------------------
# TEST 1 — the complete RLM path, driven by the real router
# ---------------------------------------------------------------------------

def test_complete_rlm_path_executes_every_stage():

    orchestrator, handler, synthesizer, child_log = build_pipeline()

    agent = NanoCodeAgent(
        console_trace=False,
        router=RLMRouter(),
        rlm_orchestrator=orchestrator,
    )

    answer = agent.run(COMPLEX_TASK)

    # 1. The real router chose the RLM path.
    assert agent.last_route_decision.strategy == STRATEGY_RLM

    # 2-3. Orchestration ran and decomposed into multiple children.
    child_tasks = [child.task for child in orchestrator.last_child_tasks]

    assert len(child_tasks) > 1

    # 4-5. Every child was submitted to the runtime, which invoked the handler.
    assert [context.task for context in handler.contexts] == child_tasks

    # 6. The children really executed: the child agent recorded each task.
    assert child_log == child_tasks

    # 7. Child results were collected.
    assert len(handler.results) == len(child_tasks)
    assert all(result.success for result in handler.results)

    # 8. The existing synthesizer received exactly those results.
    assert len(synthesizer.batches) == 1
    assert synthesizer.batches[0] == handler.results

    # 9. The final result succeeded.
    assert orchestrator.last_result.success is True

    # 10. The answer is built from what the children actually returned.
    for child_answer in CHILD_ANSWERS[: len(child_tasks)]:
        assert child_answer in answer


def test_runtime_budget_counters_record_the_executed_children():

    orchestrator, handler, _, _ = build_pipeline()

    orchestrator.run(COMPLEX_TASK)

    runtime = orchestrator.runtime

    expected = len(orchestrator.last_child_tasks)

    # The runtime's own counters are independent evidence that the calls went
    # through RLMRuntime rather than around it.
    assert runtime.budget.children_created == expected
    assert runtime.budget.iterations == expected
    assert len(handler.contexts) == expected


# ---------------------------------------------------------------------------
# TEST 2 — child context derivation and metadata isolation
# ---------------------------------------------------------------------------

def test_children_receive_derived_context_from_the_parent():

    orchestrator, handler, _, _ = build_pipeline()

    parent = RLMContext(
        task=COMPLEX_TASK,
        content="Important project context",
        depth=0,
        metadata={"request_id": "test-123", "source": "integration-test"},
    )

    children = DeterministicRLMDecomposer().decompose(parent.task, parent)

    orchestrator.runtime.call_and_synthesize(
        parent=parent,
        tasks=children,
        synthesizer=orchestrator.synthesizer,
    )

    assert len(handler.contexts) == len(children)

    for context in handler.contexts:
        assert context.depth == parent.depth + 1
        assert context.metadata == parent.metadata


def test_child_metadata_is_isolated_from_the_parent_and_siblings():

    orchestrator, handler, _, _ = build_pipeline()

    parent = RLMContext(
        task=COMPLEX_TASK,
        metadata={"request_id": "test-123", "source": "integration-test"},
    )

    children = DeterministicRLMDecomposer().decompose(parent.task, parent)

    orchestrator.runtime.call_and_synthesize(
        parent=parent,
        tasks=children,
        synthesizer=orchestrator.synthesizer,
    )

    first, second = handler.contexts[0], handler.contexts[1]

    first.metadata["mutated"] = True

    assert "mutated" not in parent.metadata
    assert "mutated" not in second.metadata


# ---------------------------------------------------------------------------
# TEST 3 — multiple, distinct child executions
# ---------------------------------------------------------------------------

def test_multiple_distinct_children_actually_execute():

    orchestrator, handler, _, child_log = build_pipeline()

    orchestrator.run(COMPLEX_TASK)

    executed = [context.task for context in handler.contexts]

    assert len(executed) > 1
    assert len(set(executed)) == len(executed)

    # Proven by the execution record, not by the decomposer's return value.
    assert child_log == executed


def test_every_child_carries_the_parent_task_as_content():

    orchestrator, handler, _, _ = build_pipeline()

    orchestrator.run(COMPLEX_TASK)

    assert all(context.content == COMPLEX_TASK for context in handler.contexts)


# ---------------------------------------------------------------------------
# TEST 4 — one failing child does not abort its siblings
# ---------------------------------------------------------------------------

def test_a_failing_child_does_not_abort_the_others():

    log: list[str] = []

    orchestrator, handler, synthesizer, _ = build_pipeline(
        agent=SelectiveFailureAgent(fail_marker="configuration", log=log),
    )

    answer = orchestrator.run(COMPLEX_TASK)

    expected = [child.task for child in orchestrator.last_child_tasks]

    assert len(expected) == 3

    # Children 1, 2 and 3 all ran, including the ones after the failure.
    assert log == expected
    assert [context.task for context in handler.contexts] == expected

    successes = [result.success for result in handler.results]

    assert successes.count(True) == 2
    assert successes.count(False) == 1

    # The failed child is represented with the existing RLMResult shape.
    failed = [result for result in handler.results if not result.success][0]

    assert failed.answer == ""
    assert "child exploded" in failed.metadata["error"]

    # Synthesis still succeeded from the surviving children.
    assert len(synthesizer.batches[0]) == 3
    assert orchestrator.last_result.success is True
    assert answer


# ---------------------------------------------------------------------------
# TEST 5 — all children failing uses the existing failure representation
# ---------------------------------------------------------------------------

def test_all_children_failing_keeps_the_existing_failure_contract():

    class AlwaysFailingAgent:
        log: list[str] = []

        def run(self, task: str) -> str:
            AlwaysFailingAgent.log.append(task)
            raise RuntimeError("child exploded")

    AlwaysFailingAgent.log = []

    orchestrator, handler, synthesizer, _ = build_pipeline(
        agent=AlwaysFailingAgent(),
    )

    answer = orchestrator.run(COMPLEX_TASK)

    expected = [child.task for child in orchestrator.last_child_tasks]

    # Every child was still attempted.
    assert AlwaysFailingAgent.log == expected
    assert len(synthesizer.batches[0]) == len(expected)

    result = orchestrator.last_result

    assert isinstance(result, RLMResult)
    assert result.success is False
    assert result.answer == ""
    assert result.children_created == len(expected)

    # The caller still gets a string, not an exception.
    assert isinstance(answer, str)
    assert COMPLEX_TASK in answer


# ---------------------------------------------------------------------------
# TEST 6 — empty decomposition fails cleanly and executes nothing
# ---------------------------------------------------------------------------

def test_empty_decomposition_fails_cleanly():

    orchestrator, handler, synthesizer, child_log = build_pipeline()

    orchestrator.decomposer = EmptyDecomposer()

    answer = orchestrator.run(COMPLEX_TASK)

    # No child work happened at any layer.
    assert handler.contexts == []
    assert child_log == []
    assert orchestrator.runtime.budget.children_created == 0
    assert orchestrator.runtime.budget.iterations == 0

    # Synthesis was called once, with a valid (empty) batch.
    assert synthesizer.batches == [[]]

    result = orchestrator.last_result

    assert result.success is False
    assert result.answer == ""

    assert isinstance(answer, str)
    assert COMPLEX_TASK in answer


# ---------------------------------------------------------------------------
# TEST 7 — the runtime budget is the authority on child creation
# ---------------------------------------------------------------------------

def test_child_budget_is_enforced_by_the_runtime():

    orchestrator, handler, _, child_log = build_pipeline(
        budget=RLMBudget(max_depth=2, max_children=1, max_iterations=10),
    )

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.run(COMPLEX_TASK)

    # The decomposer proposed more children than the budget allows; exactly
    # one was created, and no extra child slipped past the limit.
    assert len(orchestrator.last_child_tasks) > 1
    assert orchestrator.runtime.budget.children_created == 1
    assert len(handler.contexts) == 1
    assert len(child_log) == 1


def test_iteration_budget_is_enforced_by_the_runtime():

    orchestrator, handler, _, _ = build_pipeline(
        budget=RLMBudget(max_depth=2, max_children=5, max_iterations=2),
    )

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.run(COMPLEX_TASK)

    assert orchestrator.runtime.budget.iterations == 2
    assert len(handler.contexts) == 2


# ---------------------------------------------------------------------------
# TEST 8 — max depth is enforced by the runtime
# ---------------------------------------------------------------------------

def test_max_depth_blocks_child_creation():

    orchestrator, handler, _, child_log = build_pipeline(
        budget=RLMBudget(max_depth=2, max_children=5, max_iterations=10),
    )

    # A parent already at the maximum depth: its children would be depth 3.
    parent = RLMContext(task=COMPLEX_TASK, depth=2)

    children = DeterministicRLMDecomposer().decompose(parent.task, parent)

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.runtime.call_and_synthesize(
            parent=parent,
            tasks=children,
            synthesizer=orchestrator.synthesizer,
        )

    assert handler.contexts == []
    assert child_log == []
    assert orchestrator.runtime.budget.children_created == 0


def test_children_run_one_level_below_the_parent():

    orchestrator, handler, _, _ = build_pipeline()

    orchestrator.run(COMPLEX_TASK)

    assert handler.contexts
    assert all(context.depth == 1 for context in handler.contexts)
    assert all(result.depth == 1 for result in handler.results)


# ---------------------------------------------------------------------------
# TEST 9 — the normal path never enters RLM
# ---------------------------------------------------------------------------

def test_simple_task_takes_the_normal_path_and_skips_rlm():

    orchestrator, handler, synthesizer, child_log = build_pipeline()

    calls: list[str] = []

    class RecordingStage:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, state, **kwargs):
            calls.append(self.name)

            if self.name == "executor":
                from agent.state import AgentStatus

                state.status = AgentStatus.COMPLETED
                state.final_response = "normal answer"

        def evaluate(self, state):
            calls.append("evaluator")
            return type("Evaluation", (), {"success": True, "reason": "ok"})()

        def reflect(self, state, evaluation):
            calls.append("reflector")
            return type(
                "Reflection",
                (),
                {"should_improve": False, "diagnosis": "", "improvement": ""},
            )()

    agent = NanoCodeAgent(
        console_trace=False,
        router=RLMRouter(),
        rlm_orchestrator=orchestrator,
    )

    agent.planner = RecordingStage("planner")
    agent.executor = RecordingStage("executor")
    agent.evaluator = RecordingStage("evaluator")
    agent.reflector = RecordingStage("reflector")

    answer = agent.run(SIMPLE_TASK)

    assert agent.last_route_decision.strategy == STRATEGY_NORMAL
    assert answer == "normal answer"

    # The normal pipeline ran, unchanged.
    assert calls == ["planner", "executor", "evaluator", "reflector"]

    # Nothing in the RLM stack was touched.
    assert orchestrator.last_child_tasks == []
    assert orchestrator.last_result is None
    assert handler.contexts == []
    assert synthesizer.batches == []
    assert child_log == []


# ---------------------------------------------------------------------------
# TEST 10-11 — the routing decision drives RLM, and the task is not rewritten
# ---------------------------------------------------------------------------

def test_routing_decision_is_the_reason_rlm_runs():

    router = RLMRouter()

    decision = router.decide(COMPLEX_TASK)

    assert decision.strategy == STRATEGY_RLM
    assert decision.score >= router.threshold

    orchestrator, _, _, _ = build_pipeline()

    received: list[str] = []

    original_run = orchestrator.run

    def recording_run(task: str) -> str:
        received.append(task)
        return original_run(task)

    orchestrator.run = recording_run

    agent = NanoCodeAgent(
        console_trace=False,
        router=router,
        rlm_orchestrator=orchestrator,
    )

    agent.run(COMPLEX_TASK)

    # The router hands the task over verbatim; it never rewrites it.
    assert received == [COMPLEX_TASK]


def test_original_task_survives_to_child_execution():

    orchestrator, handler, _, _ = build_pipeline()

    agent = NanoCodeAgent(
        console_trace=False,
        router=RLMRouter(),
        rlm_orchestrator=orchestrator,
    )

    agent.run(COMPLEX_TASK)

    # The generated child task never replaces the original: it travels beside
    # it as the child's content.
    for context in handler.contexts:
        assert context.content == COMPLEX_TASK
        assert context.task != COMPLEX_TASK

    assert all(
        child.content == COMPLEX_TASK
        for child in orchestrator.last_child_tasks
    )


# ---------------------------------------------------------------------------
# TEST 12 — synthesis of deterministic child results
# ---------------------------------------------------------------------------

def test_synthesis_combines_the_deterministic_child_results():

    orchestrator, handler, synthesizer, _ = build_pipeline(answers=CHILD_ANSWERS)

    answer = orchestrator.run(COMPLEX_TASK)

    batch = synthesizer.batches[0]

    assert [result.answer for result in batch] == CHILD_ANSWERS

    # The existing synthesizer joins successful answers into RLMResult.answer;
    # no second synthesis implementation is involved.
    assert orchestrator.last_result.answer == "\n\n".join(CHILD_ANSWERS)

    # The caller receives the evidence report built from those same answers
    # (Phase 8), and every child finding survives into it.
    assert answer == orchestrator.last_result.metadata["report"]

    for child_answer in CHILD_ANSWERS:
        assert child_answer in answer

    assert orchestrator.last_result.success is True
    assert orchestrator.last_result.children_created == len(CHILD_ANSWERS)


# ---------------------------------------------------------------------------
# TEST 13 — the final result keeps the existing contract
# ---------------------------------------------------------------------------

def test_final_result_uses_the_existing_result_contract():

    orchestrator, _, _, _ = build_pipeline()

    answer = orchestrator.run(COMPLEX_TASK)

    result = orchestrator.last_result

    assert isinstance(result, RLMResult)
    assert result.success is True
    assert isinstance(result.answer, str)
    assert result.answer
    assert result.depth == 1
    assert result.children_created == len(orchestrator.last_child_tasks)
    assert isinstance(result.metadata, dict)

    # NanoCode callers receive a plain string: the evidence report, which is
    # derived from result.answer rather than replacing it.
    assert isinstance(answer, str)
    assert answer == result.metadata["report"]
    assert result.answer


# ---------------------------------------------------------------------------
# TEST 15 — observation structure, verified without any network access
# ---------------------------------------------------------------------------

class SpanRecorder:
    """A stand-in Langfuse client that records observation names and nesting."""

    def __init__(self) -> None:
        self.events: list[tuple[int, str]] = []
        self._stack: list[str] = []

    def flush(self) -> None:
        """No-op: nothing is ever sent anywhere."""

    def start_as_current_observation(self, *, as_type, name, input=None):
        recorder = self

        class _Span:
            def __enter__(self_inner):
                recorder.events.append((len(recorder._stack), name))
                recorder._stack.append(name)
                return self_inner

            def __exit__(self_inner, *exc):
                recorder._stack.pop()
                return False

            def update(self_inner, **kwargs):
                return None

        return _Span()


def test_rlm_execution_is_observed_under_the_nanocode_run_trace():

    recorder = SpanRecorder()

    orchestrator, _, _, _ = build_pipeline()

    orchestrator.langfuse = recorder
    orchestrator.runtime.langfuse = recorder

    agent = NanoCodeAgent(
        console_trace=False,
        router=RLMRouter(),
        rlm_orchestrator=orchestrator,
    )

    agent.langfuse = recorder

    agent.run(COMPLEX_TASK)

    names = [name for _, name in recorder.events]

    assert names[0] == "nanocode-run"

    # Everything else is nested inside the single root trace: no child gets a
    # trace of its own.
    assert all(depth > 0 for depth, _ in recorder.events[1:])
    assert names.count("nanocode-run") == 1

    assert "routing-decision" in names
    assert "rlm-decomposition" in names
    assert "rlm-synthesis" in names

    child_count = names.count("rlm-child")

    assert child_count == len(orchestrator.last_child_tasks)
    assert child_count > 1

    # Conceptual ordering: route, then decompose, then children, then synthesis.
    assert names.index("routing-decision") < names.index("rlm-decomposition")
    assert names.index("rlm-decomposition") < names.index("rlm-child")
    assert names.index("rlm-child") < names.index("rlm-synthesis")
