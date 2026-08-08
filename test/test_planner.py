from agent.planner import Planner
from agent.state import AgentState
from models.config import AgentConfig


state = AgentState(
    task="Create a Python program that reads a CSV file and calculates the average of a column.",
    messages=[],
    tools=[],
    config=AgentConfig(),
)

planner = Planner()

planner.run(state)

print("\nGenerated Plan:")
print("----------------")

for i, step in enumerate(state.plan, start=1):
    print(f"{i}. {step}")

assert len(state.plan) > 0

print("\n✅ Planner test passed")