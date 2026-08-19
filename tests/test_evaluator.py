from agent.evaluator import Evaluator
from agent.state import AgentState, AgentStatus
from models.config import AgentConfig


def make_state(
    status: AgentStatus,
    final_response: str = "",
) -> AgentState:
    """Create a minimal AgentState for evaluator tests."""

    return AgentState(
        task="Test task",
        messages=[],
        tools=[],
        config=AgentConfig(),
        status=status,
        final_response=final_response,
    )


evaluator = Evaluator()


# ---------------------------------------------------------
# SUCCESS
# ---------------------------------------------------------

state = make_state(
    AgentStatus.COMPLETED,
    "Task completed successfully.",
)

result = evaluator.evaluate(state)

assert result.success is True
assert result.score == 1.0

print("✅ Success evaluation passed")


# ---------------------------------------------------------
# FAILURE
# ---------------------------------------------------------

state = make_state(
    AgentStatus.FAILED,
)

result = evaluator.evaluate(state)

assert result.success is False
assert result.score == 0.0

print("✅ Failure evaluation passed")


# ---------------------------------------------------------
# EMPTY FINAL RESPONSE
# ---------------------------------------------------------

state = make_state(
    AgentStatus.COMPLETED,
    "",
)

result = evaluator.evaluate(state)

assert result.success is False
assert result.score == 0.0

print("✅ Empty-response evaluation passed")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n✅ All evaluator tests passed")