from agent.memory import Experience
from agent.planner import Planner


planner = Planner()


experience = Experience(
    task="Fix Python unit test",
    diagnosis="The agent modified code before inspecting the traceback.",
    improvement="Inspect the failing test and traceback first.",
    success=False,
)


prompt = planner._build_user_prompt(
    "Fix this failing Python test.",
    [experience],
)


# ---------------------------------------------------------
# MEMORY CONTEXT
# ---------------------------------------------------------

assert "Relevant experience from previous executions:" in prompt
assert "Fix Python unit test" in prompt
assert "The agent modified code before inspecting the traceback." in prompt
assert "Inspect the failing test and traceback first." in prompt


# ---------------------------------------------------------
# CURRENT TASK
# ---------------------------------------------------------

assert "Current task:" in prompt
assert "Fix this failing Python test." in prompt


print("✅ Planner memory context test passed")


# ---------------------------------------------------------
# NO MEMORY
# ---------------------------------------------------------

prompt_without_memory = planner._build_user_prompt(
    "What is Python?",
    [],
)

assert prompt_without_memory == "What is Python?"

print("✅ Planner empty memory context test passed")


print("\n✅ All planner memory tests passed")