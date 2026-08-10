from agent.evaluator import EvaluationResult
from agent.reflector import Reflector
from agent.state import AgentState, AgentStatus
from agent.tracer import Tracer
from models.config import AgentConfig


def make_state() -> AgentState:
    """Create a minimal state for reflection tests."""

    return AgentState(
        task="Test task",
        messages=[],
        tools=[],
        config=AgentConfig(),
        status=AgentStatus.COMPLETED,
        final_response="Test response",
    )


tracer = Tracer()
reflector = Reflector(tracer=tracer)
state = make_state()


# ---------------------------------------------------------
# SUCCESSFUL EVALUATION
# ---------------------------------------------------------

evaluation = EvaluationResult(
    success=True,
    score=1.0,
    reason="Task completed successfully.",
)

result = reflector.reflect(
    state,
    evaluation,
)

assert result.should_improve is False
assert result.diagnosis == "The task was completed successfully."
assert result.improvement == "No improvement required."

print("✅ Successful reflection passed")


# ---------------------------------------------------------
# FAILED EVALUATION
# ---------------------------------------------------------

evaluation = EvaluationResult(
    success=False,
    score=0.0,
    reason="Agent execution failed.",
)

result = reflector.reflect(
    state,
    evaluation,
)

assert result.should_improve is True
assert result.diagnosis == "Agent execution failed."
assert result.improvement

print("✅ Failed reflection passed")


# ---------------------------------------------------------
# TRACE TEST
# ---------------------------------------------------------

events = tracer.get_events()

reflection_events = [
    event
    for event in events
    if event.name == "reflection.completed"
]

assert len(reflection_events) == 2

# Check the latest reflection event
event = reflection_events[-1]

assert event.component == "reflector"
assert event.data["should_improve"] is True
assert event.data["diagnosis"] == "Agent execution failed."

print("✅ Reflection tracing passed")


print("\n✅ All reflector tests passed")