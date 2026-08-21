"""Tests for RLM child event propagation to the CLI event path.

Deterministic: fake agents with real Tracers, a recording event sink, and the
real TerminalRenderer captured via capsys. No OpenRouter, no network.
"""

import ast

import pytest

from agent.agent import NanoCodeAgent
from agent.state import AgentStatus
from agent.tracer import Tracer
from cli.renderer import TerminalRenderer
from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.nanocode_handler import NanoCodeCallHandler
from rlm.orchestrator import RLMOrchestrator
from rlm.router import STRATEGY_NORMAL, STRATEGY_RLM, RLMRouter
from rlm.runtime import RLMRuntime


COMPLEX_TASK = "Find the authentication bug across the project."
SIMPLE_TASK = "What is Python?"


class TracingChildAgent:
    """A child agent that emits the real events a NanoCode child emits.

    It has its own Tracer, exactly like NanoCodeAgent, so the propagation
    being tested is the real mechanism and not a stand-in for it.
    """

    def __init__(self, tools=("grep", "read_file"), fail=False) -> None:
        self.tracer = Tracer(enabled=True, console=False, on_event=None)
        self.tools = tools
        self.fail = fail

    def run(self, task: str) -> str:
        self.tracer.record("planner.started", component="planner", task=task)
        self.tracer.record("planner.completed", component="planner", task=task)

        self.tracer.record("executor.started", component="agent", task=task)

        for tool in self.tools:
            self.tracer.record("tool.started", component="executor", tool=tool)
            self.tracer.record("tool.completed", component="executor", tool=tool)

        if self.fail:
            self.tracer.record("executor.failed", component="agent", error="boom")
            raise RuntimeError("child exploded")

        self.tracer.record("executor.completed", component="agent", task=task)
        self.tracer.record("evaluator.completed", component="evaluator", success=True)
        self.tracer.record(
            "reflector.completed", component="reflector", should_improve=False
        )

        return f"answer for {task}"


def build_recording_agent(agent_factory, budget=None):
    """A parent agent whose events are all captured in a list."""

    events = []

    agent = NanoCodeAgent(
        console_trace=False,
        trace_callback=events.append,
        router=RLMRouter(),
    )

    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=NanoCodeCallHandler(
                agent_factory=agent_factory,
                rate_limit_delay=0.0,
            ),
            budget=budget or RLMBudget(max_depth=2, max_children=5, max_iterations=10),
        ),
        tracer=agent.tracer,
    )

    agent.rlm_orchestrator = orchestrator

    return agent, orchestrator, events


def names(events) -> list[str]:
    return [event.name for event in events]


def child_events(events):
    return [event for event in events if event.data.get("rlm_child")]


# ---------------------------------------------------------------------------
# 1. Normal NanoCode events still reach the CLI
# ---------------------------------------------------------------------------

def test_normal_agent_events_still_reach_the_callback():

    events = []

    agent = NanoCodeAgent(console_trace=False, trace_callback=events.append)

    class Stage:
        def __init__(self, name):
            self.name = name

        def run(self, state, **kwargs):
            agent.tracer.record(f"{self.name}.started", component=self.name)

            if self.name == "executor":
                state.status = AgentStatus.COMPLETED
                state.final_response = "normal answer"

        def evaluate(self, state):
            agent.tracer.record("evaluator.completed", component="evaluator", success=True)
            return type("Evaluation", (), {"success": True, "reason": "ok"})()

        def reflect(self, state, evaluation):
            return type(
                "Reflection",
                (),
                {"should_improve": False, "diagnosis": "", "improvement": ""},
            )()

    agent.planner = Stage("planner")
    agent.executor = Stage("executor")
    agent.evaluator = Stage("evaluator")
    agent.reflector = Stage("reflector")

    agent.run(SIMPLE_TASK)

    assert "planner.started" in names(events)
    assert "executor.started" in names(events)
    assert "evaluator.completed" in names(events)

    # No RLM identity leaks onto normal events.
    assert child_events(events) == []


# ---------------------------------------------------------------------------
# 2-5. Child planner / executor / tool events reach the parent sink
# ---------------------------------------------------------------------------

@pytest.fixture
def rlm_run():
    agent, orchestrator, events = build_recording_agent(lambda: TracingChildAgent())

    agent.run(COMPLEX_TASK)

    return agent, orchestrator, events


def test_child_planner_started_reaches_the_parent_sink(rlm_run):

    _, _, events = rlm_run

    planner_started = [
        event
        for event in child_events(events)
        if event.name == "planner.started"
    ]

    assert planner_started
    assert planner_started[0].component == "planner"


def test_child_planner_completed_reaches_the_parent_sink(rlm_run):

    _, _, events = rlm_run

    assert any(
        event.name == "planner.completed" for event in child_events(events)
    )


def test_child_executor_events_reach_the_parent_sink(rlm_run):

    _, _, events = rlm_run

    child_names = names(child_events(events))

    assert "executor.started" in child_names
    assert "executor.completed" in child_names


def test_child_tool_events_reach_the_parent_sink(rlm_run):

    _, _, events = rlm_run

    tools = [
        event.data.get("tool")
        for event in child_events(events)
        if event.name == "tool.started"
    ]

    assert "grep" in tools
    assert "read_file" in tools


def test_child_evaluation_and_reflection_reach_the_parent_sink(rlm_run):

    _, _, events = rlm_run

    child_names = names(child_events(events))

    assert "evaluator.completed" in child_names
    assert "reflector.completed" in child_names


# ---------------------------------------------------------------------------
# 6. Child completion and failure are propagated
# ---------------------------------------------------------------------------

def test_child_completion_is_propagated(rlm_run):

    _, orchestrator, events = rlm_run

    completed = [
        event for event in events if event.name == "rlm.child.completed"
    ]

    assert len(completed) == len(orchestrator.last_child_tasks)


def test_child_failure_is_propagated():

    agent, orchestrator, events = build_recording_agent(
        lambda: TracingChildAgent(fail=True)
    )

    agent.run(COMPLEX_TASK)

    failures = [event for event in events if event.name == "rlm.child.failed"]

    assert len(failures) == len(orchestrator.last_child_tasks)
    assert failures[0].data["error_type"] == "error"

    # The child's own failure event came through too.
    assert "executor.failed" in names(child_events(events))


def test_rlm_lifecycle_events_are_emitted(rlm_run):

    _, orchestrator, events = rlm_run

    event_names = names(events)

    assert "rlm.started" in event_names
    assert "rlm.decomposition.completed" in event_names
    assert "rlm.child.started" in event_names
    assert "rlm.synthesis.completed" in event_names

    decomposed = [
        event
        for event in events
        if event.name == "rlm.decomposition.completed"
    ][0]

    assert decomposed.data["children"] == len(orchestrator.last_child_tasks)


# ---------------------------------------------------------------------------
# 7-8. Children are distinguishable and identity is preserved
# ---------------------------------------------------------------------------

def test_multiple_children_produce_distinguishable_events(rlm_run):

    _, orchestrator, events = rlm_run

    indexes = {event.data["rlm_child"] for event in child_events(events)}

    assert indexes == set(range(1, len(orchestrator.last_child_tasks) + 1))


def test_child_events_preserve_child_identity(rlm_run):

    _, orchestrator, events = rlm_run

    tasks = [child.task for child in orchestrator.last_child_tasks]

    for event in child_events(events):
        assert event.data["rlm_child_count"] == len(tasks)
        assert event.data["rlm_child_task"] in tasks
        assert event.data["depth"] == 1


def test_child_identity_matches_the_child_task(rlm_run):

    _, orchestrator, events = rlm_run

    tasks = [child.task for child in orchestrator.last_child_tasks]

    for event in child_events(events):
        expected = tasks[event.data["rlm_child"] - 1]

        assert event.data["rlm_child_task"] == expected


# ---------------------------------------------------------------------------
# 9. Ordering follows execution order
# ---------------------------------------------------------------------------

def test_events_are_delivered_in_execution_order(rlm_run):

    _, _, events = rlm_run

    event_names = names(events)

    assert event_names.index("rlm.started") < event_names.index(
        "rlm.decomposition.completed"
    )
    assert event_names.index("rlm.decomposition.completed") < event_names.index(
        "rlm.child.started"
    )
    assert event_names.index("rlm.child.started") < event_names.index(
        "rlm.synthesis.completed"
    )

    # Within a child: plan, then execute, then tools.
    first_child = [
        event for event in child_events(events) if event.data["rlm_child"] == 1
    ]

    first_names = names(first_child)

    assert first_names.index("planner.started") < first_names.index(
        "executor.started"
    )
    assert first_names.index("executor.started") < first_names.index(
        "tool.started"
    )

    # Children run in order: every child 1 event precedes every child 2 event.
    indexes = [event.data["rlm_child"] for event in child_events(events)]

    assert indexes == sorted(indexes)


# ---------------------------------------------------------------------------
# 10. Normal (non-RLM) behavior is unchanged
# ---------------------------------------------------------------------------

def test_normal_route_does_not_emit_rlm_events():

    events = []

    agent = NanoCodeAgent(
        console_trace=False,
        trace_callback=events.append,
        router=RLMRouter(),
    )

    class Stage:
        def __init__(self, name):
            self.name = name

        def run(self, state, **kwargs):
            if self.name == "executor":
                state.status = AgentStatus.COMPLETED
                state.final_response = "normal answer"

        def evaluate(self, state):
            return type("Evaluation", (), {"success": True, "reason": "ok"})()

        def reflect(self, state, evaluation):
            return type(
                "Reflection",
                (),
                {"should_improve": False, "diagnosis": "", "improvement": ""},
            )()

    agent.planner = Stage("planner")
    agent.executor = Stage("executor")
    agent.evaluator = Stage("evaluator")
    agent.reflector = Stage("reflector")

    answer = agent.run(SIMPLE_TASK)

    assert agent.last_route_decision.strategy == STRATEGY_NORMAL
    assert answer == "normal answer"

    assert not any(name.startswith("rlm.") for name in names(events))


def test_a_handler_without_forwarding_still_works():

    # No tracer attached: the RLM path must behave exactly as before.
    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=NanoCodeCallHandler(
                agent_factory=lambda: TracingChildAgent(),
            ),
            budget=RLMBudget(max_children=5, max_iterations=10),
        ),
    )

    answer = orchestrator.run(COMPLEX_TASK)

    assert answer
    assert orchestrator.last_result.success is True


def test_a_child_agent_without_a_tracer_is_tolerated():

    class PlainAgent:
        def run(self, task: str) -> str:
            return f"answer for {task}"

    agent, orchestrator, events = build_recording_agent(lambda: PlainAgent())

    agent.run(COMPLEX_TASK)

    # No child events (the agent has none to give), but the RLM lifecycle
    # events still arrive and nothing crashes.
    assert child_events(events) == []
    assert "rlm.child.started" in names(events)
    assert orchestrator.last_result.success is True


# ---------------------------------------------------------------------------
# 11-12. No stdout or renderer coupling in the RLM layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_path",
    [
        "rlm/runtime.py",
        "rlm/nanocode_handler.py",
        "rlm/orchestrator.py",
        "rlm/synthesizer.py",
        "rlm/decomposer.py",
    ],
)
def test_rlm_modules_do_not_print(module_path):

    tree = ast.parse(open(module_path, encoding="utf-8").read())

    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert "print" not in calls


@pytest.mark.parametrize(
    "module_path",
    [
        "rlm/runtime.py",
        "rlm/nanocode_handler.py",
        "rlm/orchestrator.py",
    ],
)
def test_rlm_modules_do_not_import_the_renderer(module_path):

    source = open(module_path, encoding="utf-8").read()

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

    assert "TerminalRenderer" not in source
    assert "sys.stdout" not in source


# ---------------------------------------------------------------------------
# The real renderer, driven by the real events
# ---------------------------------------------------------------------------

def test_the_real_renderer_shows_child_activity(capsys):

    renderer = TerminalRenderer()

    agent, orchestrator, events = build_recording_agent(lambda: TracingChildAgent())

    agent.tracer.on_event = renderer.handle_trace

    agent.run(COMPLEX_TASK)

    output = capsys.readouterr().out

    assert "RLM: decomposed into 3 child tasks" in output
    assert "RLM child 1/3" in output

    # Child work is rendered through the existing trace rendering, tagged with
    # the child it came from.
    assert "[RLM child 1/3] Planning..." in output
    assert "[RLM child 1/3] Executing..." in output
    assert "[RLM child 1/3] grep" in output
    assert "[RLM child 2/3] read_file" in output

    assert "RLM: synthesis complete" in output


def test_the_renderer_marks_a_partial_run(capsys):

    renderer = TerminalRenderer()

    class FirstChildOnly:
        def __init__(self) -> None:
            self.calls = 0
            self.tracer = Tracer(enabled=True, console=False, on_event=None)

        def run(self, task: str) -> str:
            self.calls += 1

            self.tracer.record("planner.started", component="planner", task=task)

            if self.calls == 1:
                return "child one answer"

            raise RuntimeError("child exploded")

    child = FirstChildOnly()

    agent, _, _ = build_recording_agent(lambda: child)

    agent.tracer.on_event = renderer.handle_trace

    agent.run(COMPLEX_TASK)

    output = capsys.readouterr().out

    assert "RLM child 2 failed" in output
    assert "RLM: partial result -- 1 of 3 child analyses completed" in output


def test_normal_rendering_is_unprefixed(capsys):

    renderer = TerminalRenderer()

    tracer = Tracer(enabled=True, console=False, on_event=renderer.handle_trace)

    tracer.record("planner.started", component="planner")
    tracer.record("tool.started", component="executor", tool="grep")

    output = capsys.readouterr().out

    assert "● Planning..." in output
    assert "RLM child" not in output
    assert "  ↳ grep" in output
