# NanoCode --- Phase 6: Experience / Memory

## 1. Goal

Phase 6 adds an **experience and memory layer** to NanoCode.

Up to Phase 5:

```text
Task
 ↓
Plan
 ↓
Execute
 ↓
Trace
 ↓
Evaluate
 ↓
Reflect
```

Phase 6 adds:

```text
Task
 ↓
Retrieve relevant experience
 ↓
Plan
 ↓
Execute
 ↓
Evaluate
 ↓
Reflect
 ↓
Store useful experience
```

The goal is to make NanoCode benefit from lessons learned during
previous executions.

---

## 2. Why Memory Comes After Reflection

Reflection produces an actionable lesson:

```text
Execution
 ↓
Evaluation
 ↓
Reflection
 ↓
Diagnosis + Improvement
```

Without memory, that lesson disappears when the run ends.

Phase 6 changes this:

```text
Reflection
 ↓
Experience
 ↓
Memory
 ↓
Future task
 ↓
Relevant experience retrieved
```

The conceptual learning loop becomes:

```text
Attempt
 ↓
Evaluate
 ↓
Reflect
 ↓
Remember
 ↓
Future attempt
 ↓
Use previous experience
```

---

## 3. Phase 6 Scope

Phase 6 should implement:

- `Experience`
- `Memory`
- Adding experiences
- Retrieving experiences
- Clearing memory
- Basic experience selection
- Converting reflection results into experiences
- Storing useful experiences after execution
- Supplying relevant experiences to future planning
- Memory tracing
- Unit tests
- Integration tests

Do **not** initially implement:

- Vector databases
- Embeddings
- RAG
- FAISS
- Chroma
- Redis
- Cloud persistence
- Complex semantic similarity
- Automatic recursive retries

The first implementation should be deterministic and local.

---

## 4. Phase 6 Architecture

```text
                         NanoCodeAgent
                              |
                              v
                       Memory Retrieval
                              |
                              v
                           Planner
                              |
                              v
                          Executor
                              |
                 +------------+------------+
                 |                         |
                 v                         v
                LLM                       Tools
                 |                         |
                 +------------+------------+
                              |
                              v
                            Trace
                              |
                              v
                          Evaluator
                              |
                              v
                       EvaluationResult
                              |
                              v
                          Reflector
                              |
                              v
                       ReflectionResult
                              |
                              v
                      Experience Builder
                              |
                              v
                           Memory
```

The important new relationships are:

```text
Reflector → Experience → Memory
Memory → Planner
```

---

## 5. Experience

Create a structured representation of a lesson.

Suggested structure:

```python
@dataclass
class Experience:
    task: str
    diagnosis: str
    improvement: str
    success: bool
```

Example:

```python
Experience(
    task="Find the Python version",
    diagnosis="The agent made unnecessary repeated environment checks.",
    improvement="Check the active interpreter first.",
    success=True,
)
```

Another example:

```python
Experience(
    task="Fix a failing unit test",
    diagnosis="The agent modified code before inspecting the failing test.",
    improvement="Inspect the failing test and error output before making changes.",
    success=False,
)
```

---

## 6. Memory

Create:

```text
agent/memory.py
```

The initial implementation can simply hold experiences in a Python list:

```python
class Memory:

    def __init__(self):
        self.experiences = []

    def add(self, experience):
        self.experiences.append(experience)

    def get_all(self):
        return self.experiences.copy()

    def clear(self):
        self.experiences.clear()

    def count(self):
        return len(self.experiences)
```

Keep the storage abstraction simple so it can later be replaced with
SQLite or a vector store without redesigning the agent.

---

## 7. Why Not Use a Vector Database Yet?

Do not immediately add:

```text
Task
 ↓
Embedding
 ↓
Vector DB
 ↓
Similarity Search
```

First establish:

```text
Experience
 ↓
Memory
 ↓
store / retrieve
```

Later, the same interface can be backed by:

```text
Python list
 ↓
SQLite
 ↓
Vector database
```

---

## 8. Experience Creation

After reflection:

```text
EvaluationResult
       +
ReflectionResult
       +
AgentState
       ↓
Experience
```

Example:

```python
Experience(
    task=state.task,
    diagnosis=reflection.diagnosis,
    improvement=reflection.improvement,
    success=evaluation.success,
)
```

Use a small helper or method for this conversion rather than duplicating
the logic throughout the agent.

---

## 9. Which Experiences Should Be Stored?

Not every successful task needs to become a long-term lesson.

For the first implementation, use:

```text
reflection.should_improve == True
        ↓
store experience
```

This keeps memory focused on actionable lessons.

Failed executions are especially valuable:

```text
Evaluation
success = False
        ↓
Reflection
should_improve = True
        ↓
Store Experience
```

---

## 10. Basic Memory Retrieval

The first retrieval mechanism should be simple and deterministic.

Do not implement semantic embeddings yet.

A basic approach:

```text
current task
     ↓
compare words with stored tasks
     ↓
return relevant experiences
```

For example:

Stored task:

```text
Fix Python unit test
```

New task:

```text
Fix failing Python test
```

Shared keywords:

```text
fix
python
test
```

The memory system can return that experience.

---

## 11. Suggested Retrieval API

Eventually:

```python
memory.retrieve(task, limit=3)
```

Example:

```python
experiences = memory.retrieve(
    "Fix a failing Python test",
    limit=3,
)
```

The result should contain the most relevant experiences.

---

## 12. Basic Keyword Retrieval Rules

For the initial implementation:

1.  Normalize task text.
2.  Split it into words.
3.  Compare current task words with experience task words.
4.  Calculate word overlap.
5.  Ignore experiences with zero overlap.
6.  Sort by overlap score.
7.  Return up to `limit` experiences.

A simple overlap score is:

```python
overlap = len(current_words & experience_words)
```

This is intentionally primitive. The goal is to establish a retrieval
interface.

---

## 13. Memory and Planner

Current:

```text
Task
 ↓
Planner
```

Phase 6:

```text
Task
 ↓
Memory.retrieve()
 ↓
Relevant Experience
 ↓
Planner
```

The planner can receive context such as:

```text
Previous relevant experience:

- Previous task: Fix Python unit test
- Diagnosis: The agent modified code before inspecting the failing test.
- Improvement: Inspect the failing test and error output first.

Current task:
Fix the failing test in calculator.py.
```

The exact way this context enters the planner should follow the current
planner implementation.

---

## 14. Keep Planner Changes Minimal

Do not redesign the planner.

Conceptually:

```text
Planner Input
    |
    +── Current Task
    |
    +── Relevant Experience
    |
    v
Planner
    |
    v
Plan
```

Memory should provide additional context, not replace the planner.

---

## 15. Integrating Memory into NanoCodeAgent

Add:

```python
self.memory = Memory()
```

The shared architecture becomes:

```text
NanoCodeAgent
       |
       +── Tracer
       +── Planner
       +── Executor
       +── Evaluator
       +── Reflector
       └── Memory
```

The conceptual run flow becomes:

```python
state = self.create_state(task)

experiences = self.memory.retrieve(task)

# Make relevant experiences available to planning.

self.planner.run(state)

self.executor.run(state)

evaluation = self.evaluator.evaluate(state)

reflection = self.reflector.reflect(
    state,
    evaluation,
)

if reflection.should_improve:
    experience = Experience(
        task=state.task,
        diagnosis=reflection.diagnosis,
        improvement=reflection.improvement,
        success=evaluation.success,
    )

    self.memory.add(experience)

return state.final_response
```

The exact mechanism for passing `experiences` to the planner should be
determined after inspecting the current `Planner` implementation.

Do not blindly modify `state.messages` if that conflicts with the
existing architecture.

---

## 16. Memory Tracing

Memory operations should use the shared tracer.

Useful events:

```text
memory.retrieved
memory.stored
```

Example:

```text
[TRACE] ... | memory | memory.retrieved
task: Fix Python unit test
matches: 2
```

When storing:

```text
[TRACE] ... | memory | memory.stored
task: Fix Python unit test
success: False
```

This lets the execution history show how memory influenced the agent.

---

## 17. Testing Strategy

Create:

```text
test/test_memory.py
```

Test the memory layer without an LLM.

Required tests:

1.  Empty memory
2.  Add experience
3.  Retrieve all experiences
4.  Multiple experiences
5.  Clear memory
6.  Count experiences
7.  Relevant experience retrieval
8.  Irrelevant experience filtering
9.  Retrieval limit
10. Ranking by keyword overlap

---

## 18. Example Unit Tests

### Empty memory

```python
memory = Memory()

assert memory.count() == 0
assert memory.get_all() == []
```

### Add experience

```python
experience = Experience(
    task="Fix Python test",
    diagnosis="Test failure was not inspected first.",
    improvement="Inspect the failing test before modifying code.",
    success=False,
)

memory.add(experience)

assert memory.count() == 1
```

### Retrieve experience

```python
experiences = memory.get_all()

assert len(experiences) == 1
assert experiences[0].task == "Fix Python test"
```

### Keyword retrieval

Add:

```text
Fix Python test
```

Search:

```text
Fix failing Python test
```

Expected:

```text
experience returned
```

Search:

```text
Create a React website
```

Expected:

```text
experience not returned
```

### Retrieval limit

```python
results = memory.retrieve(
    "Fix Python test",
    limit=2,
)

assert len(results) <= 2
```

---

## 19. Integration Test

After standalone memory tests pass, create:

```text
test/test_agent_memory.py
```

Expected flow:

```text
NanoCodeAgent
     |
     v
Memory.retrieve()
     |
     v
Planner
     |
     v
Executor
     |
     v
Evaluator
     |
     v
Reflector
     |
     v
Memory.add()
```

The test should verify:

```text
reflection.completed
        ↓
memory.stored
```

when:

```text
reflection.should_improve == True
```

It should also verify:

```text
memory.retrieved
```

on a subsequent task when a relevant experience exists.

---

## 20. Important Limitation

Memory is initially **process-local**.

If the program exits:

```text
Memory
 ↓
Python process ends
 ↓
Memory disappears
```

This is intentional.

The first goal is to prove the architecture.

Later, memory can become persistent:

```text
Memory
 ↓
SQLite
```

or:

```text
Memory
 ↓
Vector Database
```

---

## 21. Do Not Add Persistence Yet

Do not add:

```text
memory.json
SQLite
FAISS
Chroma
Redis
PostgreSQL
```

until the in-memory implementation is working.

This keeps the architecture easy to debug.

---

## 22. Future Semantic Retrieval

Eventually:

```text
Current Task
     ↓
Embedding
     ↓
Vector Search
     ↓
Top-K Experiences
     ↓
Planner
```

This will allow semantic similarity instead of simple keyword overlap.

That belongs to a later phase.

---

## 23. Memory Quality

Memory should store **actionable lessons**, not every piece of output.

Poor memory:

```text
Task:
"What is Python?"

Experience:
"Python is a programming language."
```

Useful memory:

```text
Task:
"Fix failing tests"

Experience:
"Always inspect the failing test and traceback before modifying code."
```

The goal is to store information that can change a future decision.

---

## 24. Phase 6 Acceptance Criteria

Phase 6 is complete when:

- [ ] `Experience` exists.
- [ ] `Memory` exists.
- [ ] Experiences can be added.
- [ ] Experiences can be retrieved.
- [ ] Memory can be cleared.
- [ ] Memory count works.
- [ ] Basic relevant-task retrieval exists.
- [ ] Irrelevant experiences are filtered.
- [ ] Retrieval limit works.
- [ ] Relevant experiences are ranked.
- [ ] Reflection can produce an experience.
- [ ] Useful experiences are stored.
- [ ] Relevant experiences can be supplied to planning.
- [ ] `memory.retrieved` is traced.
- [ ] `memory.stored` is traced.
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Existing Phase 1 tests pass.
- [ ] Existing Phase 2 tests pass.
- [ ] Existing Phase 3 tests pass.
- [ ] Existing Phase 4 tests pass.
- [ ] Existing Phase 5 tests pass.
- [ ] No vector database is required.
- [ ] No automatic retry is implemented.
- [ ] No recursive self-improvement is implemented.

---

## 25. Phase 6 Deliverable

Expected structure:

```text
agent/
├── __init__.py
├── agent.py
├── state.py
├── executor.py
├── planner.py
├── tracer.py
├── evaluator.py
├── reflector.py
└── memory.py

test/
├── test_state.py
├── test_executor.py
├── test_planner.py
├── test_tracer.py
├── test_evaluator.py
├── test_agent_evaluation.py
├── test_reflector.py
├── test_memory.py
└── test_agent_memory.py
```

---

## 26. Phase 6 Architecture Checkpoint

After Phase 6:

```text
                         NanoCodeAgent
                              |
               +--------------+--------------+
               |                             |
               v                             v
            Memory                         Planner
               |                             |
               |                             v
               |                          Executor
               |                             |
               |                    +--------+--------+
               |                    |                 |
               |                    v                 v
               |                   LLM              Tools
               |                    |                 |
               |                    +--------+--------+
               |                             |
               |                             v
               |                          Evaluator
               |                             |
               |                             v
               |                          Reflector
               |                             |
               +<----------------------------+
```

The memory loop becomes:

```text
Previous Execution
       ↓
Evaluation
       ↓
Reflection
       ↓
Experience
       ↓
Memory
       ↓
Future Task
       ↓
Retrieved Experience
       ↓
Planner
```

The system can now begin learning from previous executions.

---

## 27. Phase 6 → Phase 7

Phase 7 can introduce the actual **self-improvement loop**:

```text
Task
 ↓
Retrieve Experience
 ↓
Plan
 ↓
Execute
 ↓
Evaluate
 │
 ├── SUCCESS → Done
 │
 └── FAILURE
       ↓
    Reflect
       ↓
    Store / retrieve experience
       ↓
    Improve plan
       ↓
    Retry
```

Automatic retries and recursive behavior should only be introduced after
the memory layer is stable.

---

## 28. Phase 6 Principle

> **An experience is useful only when it can influence a future
> decision.**

The core loop is:

```text
Experience
    ↓
Store
    ↓
Retrieve
    ↓
Use
    ↓
Improve future execution
```

Phase 6 should establish this loop with the simplest reliable
implementation before adding sophisticated memory infrastructure.
