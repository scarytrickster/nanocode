from agent.agent import NanoCodeAgent
from agent.evaluator import EvaluationResult


agent = NanoCodeAgent()


# ---------------------------------------------------------
# INITIAL MEMORY
# ---------------------------------------------------------

assert agent.memory.count() == 0

print("✅ Initial memory test passed")


# ---------------------------------------------------------
# FAILED EVALUATION → MEMORY
# ---------------------------------------------------------

state = agent.create_state(
    "Fix a failing Python test"
)

evaluation = EvaluationResult(
    success=False,
    score=0.0,
    reason="Agent execution failed.",
)

reflection = agent.reflector.reflect(
    state,
    evaluation,
)

assert reflection.should_improve is True

print("✅ Failed reflection test passed")


# Store the experience exactly as NanoCodeAgent does
if reflection.should_improve:
    from agent.memory import Experience

    experience = Experience(
        task=state.task,
        diagnosis=reflection.diagnosis,
        improvement=reflection.improvement,
        success=evaluation.success,
    )

    agent.memory.add(experience)

    agent.tracer.record(
        "memory.stored",
        component="memory",
        task=state.task,
        success=evaluation.success,
    )


# ---------------------------------------------------------
# VERIFY MEMORY
# ---------------------------------------------------------

assert agent.memory.count() == 1

experiences = agent.memory.get_all()

assert len(experiences) == 1
assert experiences[0].task == "Fix a failing Python test"
assert experiences[0].success is False
assert experiences[0].diagnosis == "Agent execution failed."

print("✅ Experience stored successfully")


# ---------------------------------------------------------
# VERIFY RETRIEVAL
# ---------------------------------------------------------

results = agent.memory.retrieve(
    "Fix failing Python test"
)

assert len(results) == 1
assert results[0].task == "Fix a failing Python test"

print("✅ Stored experience can be retrieved")


# ---------------------------------------------------------
# VERIFY TRACE
# ---------------------------------------------------------

events = agent.tracer.get_events()

stored_events = [
    event
    for event in events
    if event.name == "memory.stored"
]

assert len(stored_events) == 1

event = stored_events[0]

assert event.component == "memory"
assert event.data["success"] is False

print("✅ Memory storage tracing passed")


print("\n✅ All agent memory tests passed")