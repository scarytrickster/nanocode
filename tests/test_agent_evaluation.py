from agent.agent import NanoCodeAgent


agent = NanoCodeAgent()

response = agent.run(
    "What is Python?"
)

assert response
assert response.strip()

# Make sure the evaluator was actually called.
events = agent.tracer.get_events()

evaluation_events = [
    event
    for event in events
    if event.name == "evaluation.completed"
]

assert len(evaluation_events) == 1

evaluation = evaluation_events[0]

assert evaluation.component == "evaluator"
assert evaluation.data["success"] is True
assert evaluation.data["score"] == 1.0

print("✅ Agent → Evaluator integration test passed")

reflection_events = [
    event
    for event in events
    if event.name == "reflection.completed"
]

assert len(reflection_events) == 1

reflection = reflection_events[0]

assert reflection.component == "reflector"
assert reflection.data["should_improve"] is False

print("✅ Agent → Evaluator → Reflector integration test passed")