"""Tests for deterministic RLM task decomposition.

Deterministic: fake agents only. No OpenRouter call, no LangGraph, no network.
"""

import pytest

from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.decomposer import (
    DeterministicRLMDecomposer,
    RLMChildTask,
    RLMDecomposer,
)
from rlm.nanocode_handler import NanoCodeCallHandler
from rlm.orchestrator import RLMOrchestrator
from rlm.result import RLMResult
from rlm.router import STRATEGY_RLM, RLMRouter
from rlm.runtime import RLMRuntime
from rlm.synthesizer import RLMSynthesizer


COMPLEX_TASK = "Find the authentication bug across the project."


class RecordingChildAgent:
    """A stand-in for NanoCodeAgent that records the tasks it is given."""

    def __init__(self, answer: str | None = None) -> None:
        self.tasks: list[str] = []
        self.answer = answer

    def run(self, task: str) -> str:
        self.tasks.append(task)

        return self.answer if self.answer is not None else f"answer for {task}"


class ExplodingChildAgent:
    """A child that crashes on a specific task."""

    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on
        self.tasks: list[str] = []

    def run(self, task: str) -> str:
        self.tasks.append(task)

        if self.fail_on in task:
            raise RuntimeError("child exploded")

        return f"answer for {task}"


def build_runtime(
    agent_factory=None,
    budget: RLMBudget | None = None,
) -> RLMRuntime:
    """A runtime wired to fake children."""

    return RLMRuntime(
        call_handler=NanoCodeCallHandler(
            agent_factory=agent_factory or (lambda: RecordingChildAgent()),
        ),
        budget=budget or RLMBudget(max_children=6, max_iterations=12),
    )


def words(text: str) -> set[str]:
    return {word.strip(".,:").lower() for word in text.split()}


# ---------------------------------------------------------------------------
# 1-4. Child tasks are multiple, non-empty, distinct, and on-topic
# ---------------------------------------------------------------------------

def test_complex_task_produces_multiple_children():

    children = DeterministicRLMDecomposer().decompose(COMPLEX_TASK)

    assert len(children) >= 2
    assert all(isinstance(child, RLMChildTask) for child in children)


def test_child_tasks_are_non_empty():

    children = DeterministicRLMDecomposer().decompose(COMPLEX_TASK)

    for child in children:
        assert child.task.strip()
        assert child.content == COMPLEX_TASK


def test_child_tasks_are_meaningfully_different():

    children = DeterministicRLMDecomposer().decompose(COMPLEX_TASK)

    tasks = [child.task for child in children]

    # Not merely different strings: each carries a distinct instruction, so
    # the parts that are not the shared subject must differ too.
    assert len(set(tasks)) == len(tasks)

    subject = "find the authentication bug across the project"
    prefixes = [task.lower().replace(subject, "") for task in tasks]

    assert len(set(prefixes)) == len(prefixes)


def test_child_tasks_relate_to_the_original_task():

    children = DeterministicRLMDecomposer().decompose(COMPLEX_TASK)

    topic = {"authentication", "bug", "project"}

    for child in children:
        assert topic <= words(child.task)


def test_a_comparison_task_uses_its_own_strategy():

    decomposer = DeterministicRLMDecomposer()

    task = "Compare the two login implementations."

    assert decomposer.select_strategy(task).name == "comparison"

    children = decomposer.decompose(task)

    assert len(children) >= 2
    assert len({child.task for child in children}) == len(children)


# ---------------------------------------------------------------------------
# 5. Simple tasks are not fanned out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "task",
    [
        "What is Python?",
        "Fix the typo in README.",
        "Rename this variable.",
    ],
)
def test_simple_task_produces_a_single_child(task):

    children = DeterministicRLMDecomposer().decompose(task)

    assert len(children) == 1
    assert children[0].task == task.rstrip(".")


# ---------------------------------------------------------------------------
# 6-7. Determinism
# ---------------------------------------------------------------------------

def test_decomposition_is_deterministic():

    decomposer = DeterministicRLMDecomposer()

    assert decomposer.decompose(COMPLEX_TASK) == decomposer.decompose(COMPLEX_TASK)


def test_same_task_and_context_produce_the_same_children():

    context = RLMContext(task=COMPLEX_TASK, metadata={"origin": "test"})

    first = DeterministicRLMDecomposer().decompose(COMPLEX_TASK, context)
    second = DeterministicRLMDecomposer().decompose(COMPLEX_TASK, context)

    assert first == second

    # A separate instance decomposes identically: no hidden per-object state.
    assert first == DeterministicRLMDecomposer().decompose(COMPLEX_TASK, context)


# ---------------------------------------------------------------------------
# 8-9. Decomposition neither calls an LLM nor executes children
# ---------------------------------------------------------------------------

def test_decomposition_makes_no_llm_calls(monkeypatch):

    import config.settings as settings

    def explode(*args, **kwargs):
        raise AssertionError("decomposition must not call an LLM")

    monkeypatch.setattr(
        settings.client.chat.completions,
        "create",
        explode,
    )

    assert DeterministicRLMDecomposer().decompose(COMPLEX_TASK)


def test_decomposition_does_not_execute_children():

    agent = RecordingChildAgent()

    decomposer = DeterministicRLMDecomposer()

    children = decomposer.decompose(COMPLEX_TASK)

    assert children
    assert agent.tasks == []

    # The decomposer exposes descriptions only; it has no execution surface.
    assert not hasattr(decomposer, "run")
    assert not hasattr(decomposer, "call")


def test_deterministic_decomposer_implements_the_interface():

    assert issubclass(DeterministicRLMDecomposer, RLMDecomposer)

    # The interface itself is not runnable, so an LLM-backed decomposer can
    # replace the deterministic one without touching the runtime.
    with pytest.raises(TypeError):
        RLMDecomposer()


# ---------------------------------------------------------------------------
# 10. Children execute through the existing runtime
# ---------------------------------------------------------------------------

def test_children_execute_through_the_existing_runtime():

    agent = RecordingChildAgent()

    runtime = build_runtime(agent_factory=lambda: agent)

    orchestrator = RLMOrchestrator(runtime=runtime)

    answer = orchestrator.run(COMPLEX_TASK)

    expected = [child.task for child in orchestrator.last_child_tasks]

    assert len(expected) >= 2
    assert agent.tasks == expected

    # The runtime's own budget counters are the record of what it executed.
    assert runtime.budget.children_created == len(expected)
    assert runtime.budget.iterations == len(expected)

    assert answer


# ---------------------------------------------------------------------------
# 11-12. Runtime budgets and depth remain authoritative
# ---------------------------------------------------------------------------

def test_runtime_child_budget_still_limits_execution():

    runtime = build_runtime(
        budget=RLMBudget(max_children=1, max_iterations=10),
    )

    orchestrator = RLMOrchestrator(runtime=runtime)

    # The decomposer may propose more children than the runtime allows; the
    # runtime, not the decomposer, decides how many actually run.
    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.run(COMPLEX_TASK)

    assert runtime.budget.children_created == 1


def test_runtime_max_depth_still_applies():

    runtime = build_runtime(budget=RLMBudget(max_depth=0, max_children=6))

    parent = RLMContext(task=COMPLEX_TASK)

    with pytest.raises(RuntimeError, match="budget"):
        runtime.call(parent=parent, task="child task")

    assert runtime.budget.children_created == 0


def test_default_runtime_budget_is_bounded():

    orchestrator = RLMOrchestrator(agent_factory=lambda: RecordingChildAgent())

    runtime = orchestrator._create_runtime(child_count=99)

    assert runtime.budget.max_children <= RLMBudget().max_children
    assert runtime.budget.max_depth == 2


# ---------------------------------------------------------------------------
# 13-14. Child context: depth and metadata
# ---------------------------------------------------------------------------

def test_children_run_at_child_depth():

    depths: list[int] = []

    class DepthHandler(NanoCodeCallHandler):
        def call(self, context: RLMContext) -> RLMResult:
            depths.append(context.depth)
            return super().call(context)

    runtime = RLMRuntime(
        call_handler=DepthHandler(agent_factory=lambda: RecordingChildAgent()),
        budget=RLMBudget(max_children=6, max_iterations=12),
    )

    RLMOrchestrator(runtime=runtime).run(COMPLEX_TASK)

    assert depths
    assert all(depth == 1 for depth in depths)


def test_parent_metadata_is_preserved_in_children():

    parent = RLMContext(
        task=COMPLEX_TASK,
        metadata={"run_id": "abc123", "origin": "router"},
    )

    children = DeterministicRLMDecomposer().decompose(parent.task, parent)

    contexts = [parent.child(child.task, child.content) for child in children]

    for context in contexts:
        assert context.metadata == parent.metadata
        assert context.depth == parent.depth + 1
        assert context.content == COMPLEX_TASK

    # Copied, not shared: a child cannot mutate the parent's metadata.
    contexts[0].metadata["mutated"] = True

    assert "mutated" not in parent.metadata


def test_child_content_carries_the_original_task():

    children = DeterministicRLMDecomposer().decompose(COMPLEX_TASK)

    assert all(child.content == COMPLEX_TASK for child in children)


# ---------------------------------------------------------------------------
# 15. Empty decomposition fails cleanly
# ---------------------------------------------------------------------------

class EmptyDecomposer(RLMDecomposer):
    def decompose(self, task, context=None):
        return []


def test_empty_decomposition_is_handled_cleanly():

    runtime = build_runtime()

    orchestrator = RLMOrchestrator(
        runtime=runtime,
        decomposer=EmptyDecomposer(),
    )

    answer = orchestrator.run(COMPLEX_TASK)

    assert isinstance(answer, str)
    assert COMPLEX_TASK in answer

    assert orchestrator.last_result.success is False
    assert orchestrator.last_result.answer == ""

    # Nothing was executed.
    assert runtime.budget.children_created == 0


def test_an_empty_task_decomposes_to_nothing():

    assert DeterministicRLMDecomposer().decompose("") == []
    assert DeterministicRLMDecomposer().decompose("   ") == []


# ---------------------------------------------------------------------------
# 16. A failed child does not stop its siblings
# ---------------------------------------------------------------------------

def test_a_failed_child_does_not_prevent_other_children():

    agent = ExplodingChildAgent(fail_on="configuration")

    runtime = build_runtime(agent_factory=lambda: agent)

    orchestrator = RLMOrchestrator(runtime=runtime)

    answer = orchestrator.run(COMPLEX_TASK)

    expected = [child.task for child in orchestrator.last_child_tasks]

    # Every child was attempted, including the ones after the failure.
    assert agent.tasks == expected

    assert orchestrator.last_result.success is True
    assert answer


def test_all_children_failing_uses_the_existing_failure_behavior():

    class AlwaysFailingAgent:
        def run(self, task: str) -> str:
            raise RuntimeError("child exploded")

    runtime = build_runtime(agent_factory=lambda: AlwaysFailingAgent())

    orchestrator = RLMOrchestrator(runtime=runtime)

    answer = orchestrator.run(COMPLEX_TASK)

    result = orchestrator.last_result

    assert result.success is False
    assert result.answer == ""
    assert result.children_created == len(orchestrator.last_child_tasks)

    assert COMPLEX_TASK in answer


# ---------------------------------------------------------------------------
# 17-18. Synthesis and the final result shape
# ---------------------------------------------------------------------------

def test_successful_child_results_reach_the_existing_synthesizer():

    class RecordingSynthesizer(RLMSynthesizer):
        def __init__(self) -> None:
            self.batches: list[list[RLMResult]] = []

        def synthesize(self, results):
            self.batches.append(list(results))
            return super().synthesize(results)

    synthesizer = RecordingSynthesizer()

    orchestrator = RLMOrchestrator(
        runtime=build_runtime(),
        synthesizer=synthesizer,
    )

    answer = orchestrator.run(COMPLEX_TASK)

    assert len(synthesizer.batches) == 1
    assert len(synthesizer.batches[0]) == len(orchestrator.last_child_tasks)

    for child_result in synthesizer.batches[0]:
        assert child_result.answer in answer


def test_final_rlm_result_keeps_the_existing_return_format():

    orchestrator = RLMOrchestrator(runtime=build_runtime())

    answer = orchestrator.run(COMPLEX_TASK)

    assert isinstance(answer, str)

    result = orchestrator.last_result

    assert isinstance(result, RLMResult)
    assert result.success is True
    assert result.depth == 1
    assert result.children_created == len(orchestrator.last_child_tasks)


# ---------------------------------------------------------------------------
# Router boundary: routing decides the strategy, never the children
# ---------------------------------------------------------------------------

def test_router_does_not_decompose():

    router = RLMRouter()

    decision = router.decide(COMPLEX_TASK)

    assert decision.strategy == STRATEGY_RLM

    # The router's output describes a strategy only; no child tasks anywhere.
    assert not hasattr(decision, "children")
    assert not hasattr(decision, "child_tasks")
    assert not hasattr(router, "decompose")


def test_decomposition_is_independent_of_the_router():

    import ast

    import rlm.decomposer as decomposer_module

    source = open(decomposer_module.__file__, encoding="utf-8").read()

    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    # Decomposition never reaches back into the routing layer.
    assert "rlm.router" not in imported
