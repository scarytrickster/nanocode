from agent.tracer import Tracer


tracer = Tracer()

tracer.record(
    "test.started",
    task="Tracer test",
)

tracer.record(
    "test.completed",
    success=True,
)

events = tracer.get_events()

assert len(events) == 2
assert events[0].name == "test.started"
assert events[1].name == "test.completed"
assert events[1].data["success"] is True

print("✅ Tracer test passed")