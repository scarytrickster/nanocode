from agent.tracer import Tracer
import time


tracer = Tracer()

with tracer.span(
    "test_operation",
    component="test",
):
    time.sleep(0.01)

events = tracer.get_events()

assert events[-2].name == "test_operation.started"
assert events[-1].name == "test_operation.completed"

assert events[-1].component == "test"
assert events[-1].duration_ms is not None
assert events[-1].duration_ms >= 10

print("✅ Automatic duration test passed")