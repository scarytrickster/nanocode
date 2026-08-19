from agent.memory import Experience, Memory


memory = Memory()


# ---------------------------------------------------------
# EMPTY MEMORY
# ---------------------------------------------------------

assert memory.count() == 0
assert memory.get_all() == []

print("✅ Empty memory test passed")


# ---------------------------------------------------------
# ADD EXPERIENCES
# ---------------------------------------------------------

experience_1 = Experience(
    task="Fix Python unit test",
    diagnosis="The agent modified code before inspecting the traceback.",
    improvement="Inspect the failing test and traceback first.",
    success=False,
)

experience_2 = Experience(
    task="Find Python version",
    diagnosis="The agent performed unnecessary environment checks.",
    improvement="Check the active Python interpreter first.",
    success=True,
)

experience_3 = Experience(
    task="Create React component",
    diagnosis="The component structure was unclear.",
    improvement="Break the component into smaller reusable components.",
    success=False,
)

memory.add(experience_1)
memory.add(experience_2)
memory.add(experience_3)

assert memory.count() == 3

print("✅ Add experiences test passed")


# ---------------------------------------------------------
# GET ALL
# ---------------------------------------------------------

experiences = memory.get_all()

assert len(experiences) == 3

print("✅ Get all experiences test passed")


# ---------------------------------------------------------
# RELEVANT RETRIEVAL
# ---------------------------------------------------------

results = memory.retrieve(
    "Fix failing Python test"
)

assert len(results) >= 1
assert results[0] == experience_1

print("✅ Relevant retrieval test passed")


# ---------------------------------------------------------
# IRRELEVANT RETRIEVAL
# ---------------------------------------------------------

results = memory.retrieve(
    "Build a Java Spring application"
)

assert results == []

print("✅ Irrelevant retrieval test passed")


# ---------------------------------------------------------
# RETRIEVAL LIMIT
# ---------------------------------------------------------

results = memory.retrieve(
    "Python test",
    limit=1,
)

assert len(results) <= 1

print("✅ Retrieval limit test passed")


# ---------------------------------------------------------
# RANKING
# ---------------------------------------------------------

results = memory.retrieve(
    "Fix Python test"
)

assert len(results) >= 1
assert results[0] == experience_1

print("✅ Retrieval ranking test passed")


# ---------------------------------------------------------
# CLEAR MEMORY
# ---------------------------------------------------------

memory.clear()

assert memory.count() == 0
assert memory.get_all() == []

print("✅ Clear memory test passed")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n✅ All memory tests passed")