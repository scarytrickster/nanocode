from agent.agent import NanoCodeAgent
from agent.evaluator import EvaluationResult


class FakePlanner:
    def __init__(self):
        self.received_experiences = None

    def run(self, state, experiences=None):
        self.received_experiences = experiences
        state.plan = ["Test plan"]


class FakeExecutor:
    def run(self, state):
        state.final_response = "Test response"


class FakeEvaluator:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def evaluate(self, state):
        return self.evaluation


# ---------------------------------------------------------
# CREATE AGENT
# ---------------------------------------------------------

evaluation = EvaluationResult(
    success=False,
    score=0.0,
    reason="Agent execution failed.",
)

agent = NanoCodeAgent(
    executor=FakeExecutor(),
)

# Replace the real planner/evaluator with controlled fakes.
planner = FakePlanner()
agent.planner = planner
agent.evaluator = FakeEvaluator(evaluation)


# ---------------------------------------------------------
# RUN AGENT
# ---------------------------------------------------------

response = agent.run(
    "Fix a failing Python test"
)


# ---------------------------------------------------------
# FINAL RESPONSE
# ---------------------------------------------------------

assert response == "Test response"

print("✅ Agent execution test passed")


# ---------------------------------------------------------
# VERIFY PLANNER RECEIVED MEMORY
# ---------------------------------------------------------

assert planner.received_experiences == []

print("✅ Planner received memory context")


# ---------------------------------------------------------
# VERIFY EXPERIENCE WAS STORED
# ---------------------------------------------------------

assert agent.memory.count() == 1

experience = agent.memory.get_all()[0]

assert experience.task == "Fix a failing Python test"
assert experience.diagnosis == "Agent execution failed."
assert experience.success is False
assert experience.improvement

print("✅ Experience automatically stored")


# ---------------------------------------------------------
# VERIFY MEMORY RETRIEVAL ON NEXT RUN
# ---------------------------------------------------------

planner.received_experiences = None

agent.run(
    "Fix another failing Python test"
)

assert planner.received_experiences is not None
assert len(planner.received_experiences) == 1

retrieved = planner.received_experiences[0]

assert retrieved.task == "Fix a failing Python test"

print("✅ Previous experience retrieved on next run")


# ---------------------------------------------------------
# VERIFY TRACE
# ---------------------------------------------------------

events = agent.tracer.get_events()

retrieved_events = [
    event
    for event in events
    if event.name == "memory.retrieved"
]

stored_events = [
    event
    for event in events
    if event.name == "memory.stored"
]

assert len(retrieved_events) == 2
assert len(stored_events) == 2

print("✅ Memory tracing passed")


print("\n✅ All agent memory integration tests passed")