from agent.agent import NanoCodeAgent
from agent.memory import Experience

agent = NanoCodeAgent()


# ---------------------------------------------------------
# FIRST RUN
# ---------------------------------------------------------

print("\n--- FIRST RUN ---")

response_1 = agent.run(
    "Find the Python version used in this environment."
)

assert response_1
print("✅ First real agent run completed")


# ---------------------------------------------------------
# CHECK MEMORY
# ---------------------------------------------------------

print("\n--- MEMORY AFTER FIRST RUN ---")

print(f"Stored experiences: {agent.memory.count()}")

for experience in agent.memory.get_all():
    print(f"Task: {experience.task}")
    print(f"Diagnosis: {experience.diagnosis}")
    print(f"Improvement: {experience.improvement}")
    print(f"Success: {experience.success}")

# ---------------------------------------------------------
# SEED MEMORY FOR SECOND RUN
# ---------------------------------------------------------

agent.memory.add(
    Experience(
        task="Find the Python version used in this environment",
        diagnosis="Previous execution required checking the active Python interpreter.",
        improvement="Check the active Python interpreter directly before checking other installations.",
        success=True,
    )
)

print("✅ Seed experience added")

assert agent.memory.count() == 1


# ---------------------------------------------------------
# SECOND RELATED RUN
# ---------------------------------------------------------

print("\n--- SECOND RUN ---")

response_2 = agent.run(
    "Check the Python version in this environment again."
)

assert response_2
print("✅ Second real agent run completed")


# ---------------------------------------------------------
# VERIFY MEMORY RETRIEVAL
# ---------------------------------------------------------

print("\n--- MEMORY RETRIEVAL ---")

experiences = agent.memory.retrieve(
    "Check the Python version in this environment again."
)

print(f"Retrieved experiences: {len(experiences)}")

for experience in experiences:
    print(f"Previous task: {experience.task}")
    print(f"Lesson: {experience.improvement}")


# ---------------------------------------------------------
# VERIFY TRACE
# ---------------------------------------------------------

events = agent.tracer.get_events()

retrieved_events = [
    event
    for event in events
    if event.name == "memory.retrieved"
]

assert len(retrieved_events) >= 2

print("✅ Memory retrieval tracing verified")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n✅ Real Agent → Memory E2E test completed")