"""Integration tests for the router -> NanoCode / RLM execution paths.

Everything here is deterministic: no OpenRouter, no LLM, no network.
"""

from agent.agent import NanoCodeAgent
from agent.state import AgentStatus
from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.nanocode_handler import NanoCodeCallHandler, create_nanocode_agent
from rlm.decomposer import DeterministicRLMDecomposer
from rlm.orchestrator import RLMOrchestrator
from rlm.result import RLMResult
from rlm.router import STRATEGY_NORMAL, STRATEGY_RLM, RLMRouter, RouteDecision
from rlm.runtime import RLMRuntime
from rlm.synthesizer import RLMSynthesizer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRouter:
    """Router returning a fixed strategy."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.tasks: list[str] = []

    def decide(self, task: str) -> RouteDecision:
        self.tasks.append(task)

        return RouteDecision(
            strategy=self.strategy,
            reason=f"forced {self.strategy}",
            score=1.0 if self.strategy == STRATEGY_RLM else 0.0,
            signals=["fake:signal"],
        )


class FakeOrchestrator:
    """RLM path stand-in recording the task it received."""

    def __init__(self, answer: str = "rlm answer") -> None:
        self.answer = answer
        self.tasks: list[str] = []

    def run(self, task: str) -> str:
        self.tasks.append(task)
        return self.answer


class RecordingChildAgent:
    """Child NanoCode stand-in used by NanoCodeCallHandler."""

    tasks: list[str] = []

    def run(self, task: str) -> str:
        RecordingChildAgent.tasks.append(task)
        return f"child result for {task}"


class RecordingStage:
    """Stand-in for planner / executor / evaluator / reflector stages."""

    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    def run(self, state, **kwargs):
        self.calls.append(self.name)

        if self.name == "executor":
            state.status = AgentStatus.COMPLETED
            state.final_response = "normal answer"

    def evaluate(self, state):
        self.calls.append(self.name)

        return type("Evaluation", (), {"success": True, "reason": "ok"})()

    def reflect(self, state, evaluation):
        self.calls.append(self.name)

        return type(
            "Reflection",
            (),
            {"should_improve": False, "diagnosis": "", "improvement": ""},
        )()


def build_agent(router, orchestrator=None):
    """Agent whose pipeline stages are recorded instead of hitting an LLM."""

    calls: list[str] = []

    agent = NanoCodeAgent(
        console_trace=False,
        router=router,
        rlm_orchestrator=orchestrator,
    )

    agent.planner = RecordingStage(calls, "planner")
    agent.executor = RecordingStage(calls, "executor")
    agent.evaluator = RecordingStage(calls, "evaluator")
    agent.reflector = RecordingStage(calls, "reflector")

    return agent, calls


def build_runtime(agent_factory=None):
    """Runtime wired to the real handler with a fake child agent."""

    return RLMRuntime(
        call_handler=NanoCodeCallHandler(
            agent_factory=agent_factory or (lambda: RecordingChildAgent()),
        ),
        budget=RLMBudget(max_children=5, max_iterations=10),
    )


# ---------------------------------------------------------------------------
# 1. NORMAL route
# ---------------------------------------------------------------------------

def test_normal_route_runs_nanocode_pipeline_and_skips_rlm():

    orchestrator = FakeOrchestrator()

    agent, calls = build_agent(
        router=FakeRouter(STRATEGY_NORMAL),
        orchestrator=orchestrator,
    )

    response = agent.run("Fix the typo in auth.py.")

    assert response == "normal answer"

    assert calls == ["planner", "executor", "evaluator", "reflector"]

    # RLM path untouched.
    assert orchestrator.tasks == []


def test_normal_route_records_decision():

    agent, _ = build_agent(router=FakeRouter(STRATEGY_NORMAL))

    agent.run("Fix the typo in auth.py.")

    decision = agent.last_route_decision

    assert decision is not None
    assert decision.strategy == STRATEGY_NORMAL
    assert decision.reason
    assert decision.signals == ["fake:signal"]


# ---------------------------------------------------------------------------
# 2. RLM route
# ---------------------------------------------------------------------------

def test_rlm_route_uses_rlm_path_and_skips_normal_pipeline():

    orchestrator = FakeOrchestrator(answer="synthesized answer")

    agent, calls = build_agent(
        router=FakeRouter(STRATEGY_RLM),
        orchestrator=orchestrator,
    )

    response = agent.run("Find the root cause across the project.")

    assert response == "synthesized answer"

    # Top-level NanoCode pipeline is not duplicated on the RLM path.
    assert calls == []


# ---------------------------------------------------------------------------
# 3. Original task is preserved
# ---------------------------------------------------------------------------

def test_original_task_is_preserved_through_the_rlm_path():

    task = "Find the root cause of the authentication bug across the project."

    orchestrator = FakeOrchestrator()
    router = FakeRouter(STRATEGY_RLM)

    agent, _ = build_agent(router=router, orchestrator=orchestrator)

    agent.run(task)

    assert router.tasks == [task]
    assert orchestrator.tasks == [task]

    # The decomposer keeps the original task reachable by every child.
    children = DeterministicRLMDecomposer().decompose(task)

    assert len(children) >= 2
    assert all(content == task for _, content in children)


# ---------------------------------------------------------------------------
# 4. Child execution goes through the existing RLMRuntime
# ---------------------------------------------------------------------------

def test_rlm_children_execute_through_the_existing_runtime():

    RecordingChildAgent.tasks = []

    runtime = build_runtime()

    orchestrator = RLMOrchestrator(runtime=runtime)

    answer = orchestrator.run("Analyze the entire project.")

    expected_children = len(orchestrator.last_child_tasks)

    assert expected_children >= 2

    # The budget is the runtime's own record of the calls it made.
    assert runtime.budget.children_created == expected_children
    assert runtime.budget.iterations == expected_children

    assert len(RecordingChildAgent.tasks) == expected_children

    for child_task, _ in orchestrator.last_child_tasks:
        assert child_task in RecordingChildAgent.tasks

    assert answer


def test_rlm_children_run_at_child_depth():

    depths: list[int] = []

    class DepthHandler(NanoCodeCallHandler):
        def call(self, context: RLMContext) -> RLMResult:
            depths.append(context.depth)
            return super().call(context)

    runtime = RLMRuntime(
        call_handler=DepthHandler(
            agent_factory=lambda: RecordingChildAgent(),
        ),
        budget=RLMBudget(max_children=5, max_iterations=10),
    )

    RLMOrchestrator(runtime=runtime).run("Analyze the entire project.")

    assert depths
    assert all(depth == 1 for depth in depths)


# ---------------------------------------------------------------------------
# 5. Synthesis is executed
# ---------------------------------------------------------------------------

def test_rlm_path_synthesizes_child_results():

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

    answer = orchestrator.run("Analyze the entire project.")

    assert len(synthesizer.batches) == 1
    assert len(synthesizer.batches[0]) == len(orchestrator.last_child_tasks)

    # The synthesized answer contains every child answer.
    for child_result in synthesizer.batches[0]:
        assert child_result.answer in answer


# ---------------------------------------------------------------------------
# 6. RLM result is converted to the NanoCode return format
# ---------------------------------------------------------------------------

def test_rlm_answer_is_a_string_for_nanocode_callers():

    orchestrator = RLMOrchestrator(runtime=build_runtime())

    answer = orchestrator.run("Analyze the entire project.")

    assert isinstance(answer, str)
    assert isinstance(orchestrator.last_result, RLMResult)
    assert orchestrator.last_result.success is True


def test_agent_returns_string_on_rlm_route():

    agent, calls = build_agent(
        router=FakeRouter(STRATEGY_RLM),
        orchestrator=RLMOrchestrator(runtime=build_runtime()),
    )

    response = agent.run("Analyze the entire project.")

    assert isinstance(response, str)
    assert response != ""
    assert calls == []


def test_failed_children_produce_a_string_answer():

    class FailingAgent:
        def run(self, task: str):
            return None

    orchestrator = RLMOrchestrator(
        runtime=build_runtime(agent_factory=lambda: FailingAgent()),
    )

    answer = orchestrator.run("Analyze the entire project.")

    assert isinstance(answer, str)
    assert answer != ""


# ---------------------------------------------------------------------------
# 7. Existing behavior is unchanged on the normal route
# ---------------------------------------------------------------------------

def test_normal_route_preserves_state_and_memory_behavior():

    agent, calls = build_agent(router=FakeRouter(STRATEGY_NORMAL))

    response = agent.run("What is Python?")

    assert response == "normal answer"
    assert calls == ["planner", "executor", "evaluator", "reflector"]

    # A successful run stores nothing, exactly as before.
    assert agent.memory.experiences == []

    # The task still reaches the message history.
    assert agent.messages[-1] == {
        "role": "user",
        "content": "What is Python?",
    }


def test_rlm_route_does_not_touch_state_or_memory():

    agent, _ = build_agent(
        router=FakeRouter(STRATEGY_RLM),
        orchestrator=FakeOrchestrator(),
    )

    before = len(agent.messages)

    agent.run("Analyze the entire project.")

    assert len(agent.messages) == before
    assert agent.memory.experiences == []


# ---------------------------------------------------------------------------
# Recursion guard: children must not route back into RLM.
# ---------------------------------------------------------------------------

def test_child_agents_have_routing_disabled():

    child = create_nanocode_agent()

    assert child.rlm_enabled is False
    assert child.router is None


def test_real_router_selects_the_expected_paths():

    router = RLMRouter()

    assert router.decide("Fix the typo in auth.py.").strategy == STRATEGY_NORMAL
    assert router.decide("Analyze the entire codebase.").strategy == STRATEGY_RLM
