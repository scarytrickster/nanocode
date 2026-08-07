from agent.state import AgentState
from models.config import AgentConfig

state = AgentState(
    task="Hello",
    messages=[],
    tools=[],
    config=AgentConfig(),
)

assert state.task == "Hello"
assert state.iteration == 0
assert state.status == "running"

print("✅ AgentState works")