# NanoCode --- Phase 5: Reflection

## 1. Goal

Phase 5 adds a **reflection layer** to NanoCode.

Up to Phase 4, NanoCode can:

``` text
Task
 ↓
Plan
 ↓
Execute
 ↓
Trace
 ↓
Evaluate
```

Phase 5 adds:

``` text
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

The reflector should answer:

> If the execution failed or was weak, what went wrong and what should
> be improved?

The first implementation should remain simple. **Do not add automatic
retries or recursion yet.**

------------------------------------------------------------------------

# 2. Why Reflection Comes After Evaluation

The future self-improvement loop is:

``` text
Task
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
    Improve
       ↓
    Retry
```

Evaluation tells us **whether** the attempt was successful.

Reflection tries to determine **why** and **what should change**.

Therefore:

``` text
Phase 4 → Evaluation
Phase 5 → Reflection
Phase 6 → Experience / Memory
Phase 7 → Recursive Self-Improvement
```

------------------------------------------------------------------------

# 3. Phase 5 Scope

Phase 5 should implement:

-   `ReflectionResult`
-   `Reflector`
-   Reflection based on `AgentState`
-   Reflection based on `EvaluationResult`
-   Reflection using execution trace information when useful
-   Structured reflection tracing
-   Unit tests
-   Integration with `NanoCodeAgent`

Phase 5 should NOT implement:

-   Automatic retries
-   Recursive execution
-   Long-term memory
-   Experience storage
-   Model fine-tuning
-   Self-modifying code

Those belong to later phases.

------------------------------------------------------------------------

# 4. Phase 5 Architecture

``` text
                         NanoCodeAgent
                              |
                              v
                           Planner
                              |
                              v
                         AgentState
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
```

The reflector should consume the result of the execution and evaluation.

------------------------------------------------------------------------

# 5. ReflectionResult

Create a structured result:

``` python
@dataclass
class ReflectionResult:
    should_improve: bool
    diagnosis: str
    improvement: str
```

Example:

``` python
ReflectionResult(
    should_improve=True,
    diagnosis="The agent used unnecessary tool calls before answering.",
    improvement="Use the first verified result instead of performing redundant checks.",
)
```

For a successful execution:

``` python
ReflectionResult(
    should_improve=False,
    diagnosis="The task was completed successfully.",
    improvement="No improvement required.",
)
```

The structure can be extended later.

------------------------------------------------------------------------

# 6. Reflection Responsibilities

The reflector should:

1.  Receive `AgentState`.
2.  Receive `EvaluationResult`.
3.  Inspect relevant trace events.
4.  Determine whether reflection is necessary.
5.  Produce a structured diagnosis.
6.  Produce an improvement suggestion.
7.  Record a reflection event in the tracer.

The reflector should NOT:

-   Execute tools.
-   Modify files.
-   Retry the task.
-   Modify the plan.
-   Change `AgentState` directly.
-   Recursively call `NanoCodeAgent`.

Reflection is currently an **analysis step**, not an execution step.

------------------------------------------------------------------------

# 7. Successful Execution

If:

``` python
evaluation.success is True
```

the reflector should normally return:

``` python
ReflectionResult(
    should_improve=False,
    diagnosis="The task was completed successfully.",
    improvement="No improvement required.",
)
```

Example trace:

``` text
evaluation.completed
    success: True
    score: 1.0

reflection.completed
    should_improve: False
```

Do not force the model to invent problems when the execution succeeded.

------------------------------------------------------------------------

# 8. Failed Execution

If:

``` python
evaluation.success is False
```

reflection should identify a likely problem.

For example:

``` text
Evaluation:
success = False
reason = "Agent execution failed."
```

Reflection:

``` text
should_improve = True

diagnosis =
"Execution failed before producing a valid final response."

improvement =
"Inspect the tool error and revise the execution strategy before retrying."
```

The first implementation can use deterministic rules for basic failures.

------------------------------------------------------------------------

# 9. Reflection and Trace Data

Phase 3 gave NanoCode structured trace data.

For example:

``` text
executor.started
    ↓
llm.started
    ↓
llm.completed
    ↓
tool.started
    ↓
tool.failed
    ↓
executor.failed
```

Reflection can use this information to produce:

``` text
Diagnosis:
The execution failed during tool execution.

Improvement:
Inspect the failing tool call and correct its arguments before retrying.
```

Another example:

``` text
executor.started
    ↓
llm
    ↓
tool
    ↓
llm
    ↓
tool
    ↓
llm
    ↓
completed
```

Reflection could identify:

``` text
Diagnosis:
The agent required several iterations for a simple task.

Improvement:
Reduce unnecessary tool calls and verify whether existing tool results already answer the task.
```

The exact heuristics can evolve later.

------------------------------------------------------------------------

# 10. `agent/reflector.py`

Create:

``` text
agent/reflector.py
```

Suggested initial structure:

``` python
from dataclasses import dataclass

from agent.evaluator import EvaluationResult
from agent.state import AgentState
from agent.tracer import Tracer


@dataclass
class ReflectionResult:
    should_improve: bool
    diagnosis: str
    improvement: str


class Reflector:
    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer or Tracer()

    def reflect(
        self,
        state: AgentState,
        evaluation: EvaluationResult,
    ) -> ReflectionResult:

        if evaluation.success:

            result = ReflectionResult(
                should_improve=False,
                diagnosis="The task was completed successfully.",
                improvement="No improvement required.",
            )

        else:

            result = ReflectionResult(
                should_improve=True,
                diagnosis=evaluation.reason,
                improvement=(
                    "Review the execution trace, identify the failure, "
                    "and adjust the strategy before the next attempt."
                ),
            )

        self.tracer.record(
            "reflection.completed",
            component="reflector",
            should_improve=result.should_improve,
            diagnosis=result.diagnosis,
            improvement=result.improvement,
        )

        return result
```

This is intentionally simple.

Later phases can replace deterministic reflection with an LLM-powered
reflection process.

------------------------------------------------------------------------

# 11. Why Start Deterministically?

Do not immediately make reflection another expensive LLM call.

The first goal is to establish a stable interface:

``` text
Execution
 ↓
EvaluationResult
 ↓
ReflectionResult
```

Once the interface is stable, we can replace or extend the
implementation with:

``` text
LLM Reflector
```

without changing the rest of the architecture.

This also keeps tests fast and deterministic.

------------------------------------------------------------------------

# 12. Testing Strategy

Create:

``` text
test/test_reflector.py
```

The reflector should initially be tested without an LLM.

This keeps the tests:

-   Fast
-   Deterministic
-   Cheap
-   Independent of API limits

------------------------------------------------------------------------

# 13. Test: Successful Evaluation

Create a successful `EvaluationResult`:

``` python
evaluation = EvaluationResult(
    success=True,
    score=1.0,
    reason="Task completed successfully.",
)
```

Then:

``` python
result = reflector.reflect(
    state,
    evaluation,
)
```

Assertions:

``` python
assert result.should_improve is False
assert result.diagnosis
assert result.improvement == "No improvement required."
```

Expected:

``` text
should_improve = False
```

------------------------------------------------------------------------

# 14. Test: Failed Evaluation

Create:

``` python
evaluation = EvaluationResult(
    success=False,
    score=0.0,
    reason="Agent execution failed.",
)
```

Then:

``` python
result = reflector.reflect(
    state,
    evaluation,
)
```

Assertions:

``` python
assert result.should_improve is True
assert result.diagnosis == "Agent execution failed."
assert result.improvement
```

Expected:

``` text
should_improve = True
```

------------------------------------------------------------------------

# 15. Test: Reflection Trace

The test should also verify that:

``` text
reflection.completed
```

is recorded.

Example:

``` python
events = tracer.get_events()

reflection_events = [
    event
    for event in events
    if event.name == "reflection.completed"
]

assert len(reflection_events) == 1
```

Then verify:

``` python
event = reflection_events[0]

assert event.component == "reflector"
assert event.data["should_improve"] is True
```

Adapt the event attribute access to the exact `TraceEvent`
implementation in the current project.

------------------------------------------------------------------------

# 16. Integrating Reflection into NanoCodeAgent

After `test_reflector.py` passes, connect the reflector to
`NanoCodeAgent`.

Add:

``` python
self.reflector = Reflector(
    tracer=self.tracer
)
```

The run flow becomes:

``` python
state = self.create_state(task)

self.planner.run(state)

self.executor.run(state)

evaluation = self.evaluator.evaluate(state)

reflection = self.reflector.reflect(
    state,
    evaluation,
)

return state.final_response
```

At this stage, `reflection` is only observed.

Do NOT retry automatically yet.

------------------------------------------------------------------------

# 17. Updated Agent Flow

``` text
User
 |
 v
NanoCodeAgent.run(task)
 |
 v
create_state()
 |
 v
Planner
 |
 v
Executor
 |
 +--> LLM
 |
 +--> Tools
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
Return final response
```

The important point is:

``` text
Reflection ≠ Retry
```

Reflection only analyzes the completed attempt.

------------------------------------------------------------------------

# 18. Shared Tracer

All components should continue using the same tracer:

``` text
                         Tracer
                           |
       +-------------------+-------------------+
       |                   |                   |
       v                   v                   v
    Planner             Executor           Evaluator
                                               |
                                               v
                                           Reflector
```

The complete event sequence can become:

``` text
planner.started
planner.completed

executor.started
llm.started
llm.completed
tool.started
tool.completed
llm.started
llm.completed
executor.completed

evaluation.completed

reflection.completed
```

This gives later phases a complete history of the attempt.

------------------------------------------------------------------------

# 19. Example Successful Reflection

``` text
Task:
"What is Python?"

Evaluation:
success = True
score = 1.0

Reflection:
should_improve = False

diagnosis:
"The task was completed successfully."

improvement:
"No improvement required."
```

------------------------------------------------------------------------

# 20. Example Failed Reflection

``` text
Task:
"Fix the failing test."

Evaluation:
success = False
score = 0.0
reason:
"Agent execution failed."

Reflection:
should_improve = True

diagnosis:
"Agent execution failed."

improvement:
"Review the execution trace, identify the failure, and adjust the strategy before the next attempt."
```

------------------------------------------------------------------------

# 21. Important Limitation

Phase 5 reflection is not yet a true self-improvement system.

At this stage:

``` text
Reflector
   ↓
Diagnosis
   ↓
Improvement suggestion
```

Nothing automatically changes.

The improvement is only a structured piece of information.

Later:

``` text
ReflectionResult
       ↓
Experience / Memory
       ↓
Improved planning
       ↓
Retry
```

will be introduced.

------------------------------------------------------------------------

# 22. Future LLM-Powered Reflection

A later implementation can use the LLM:

``` text
Trace
 +
Evaluation
 +
Task
 ↓
LLM Reflector
 ↓
Diagnosis
 +
Improvement
```

Prompt concept:

``` text
You are analyzing an agent execution.

Task:
{task}

Evaluation:
{evaluation}

Execution trace:
{trace}

Identify:
1. What went wrong?
2. What caused the problem?
3. What should change on the next attempt?

Return a concise structured reflection.
```

Do not implement this until the deterministic reflector is tested and
integrated.

------------------------------------------------------------------------

# 23. Phase 5 Acceptance Criteria

Phase 5 is complete when:

-   [ ] `ReflectionResult` exists.
-   [ ] `Reflector` exists.
-   [ ] Reflector accepts `AgentState`.
-   [ ] Reflector accepts `EvaluationResult`.
-   [ ] Successful evaluations produce `should_improve=False`.
-   [ ] Failed evaluations produce `should_improve=True`.
-   [ ] Diagnosis is returned.
-   [ ] Improvement suggestion is returned.
-   [ ] `reflection.completed` is recorded.
-   [ ] `test/test_reflector.py` passes.
-   [ ] Reflection is integrated into `NanoCodeAgent`.
-   [ ] Existing Phase 1 tests pass.
-   [ ] Existing Phase 2 tests pass.
-   [ ] Existing Phase 3 tests pass.
-   [ ] Existing Phase 4 tests pass.
-   [ ] Reflection does not execute tools.
-   [ ] Reflection does not retry tasks.
-   [ ] Reflection does not modify the plan.
-   [ ] Reflection does not implement recursion.
-   [ ] Reflection does not introduce long-term memory.

------------------------------------------------------------------------

# 24. Phase 5 Deliverable

Expected structure:

``` text
agent/
├── __init__.py
├── agent.py
├── state.py
├── executor.py
├── planner.py
├── tracer.py
├── evaluator.py
└── reflector.py

test/
├── test_state.py
├── test_executor.py
├── test_planner.py
├── test_tracer.py
├── test_evaluator.py
├── test_agent_evaluation.py
└── test_reflector.py
```

------------------------------------------------------------------------

# 25. Phase 5 Architecture Checkpoint

After Phase 5:

``` text
User
 |
 v
NanoCodeAgent
 |
 v
AgentState
 |
 +--> Planner
 |
 +--> Executor
       |
       +--> LLM
       |
       +--> Tools
 |
 v
Tracer
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
```

The system can now answer two questions:

``` text
1. Did the attempt succeed?
        ↓
   EvaluationResult

2. If not, what should improve?
        ↓
   ReflectionResult
```

But it still cannot automatically improve itself.

That is intentional.

------------------------------------------------------------------------

# 26. Phase 5 → Phase 6

Phase 6 will introduce **Experience / Memory**.

The flow becomes:

``` text
Execution
    ↓
Evaluation
    ↓
Reflection
    ↓
Experience
    ↓
Store useful lesson
```

For example:

``` text
Task:
"Find the Python version."

Reflection:
"The agent made unnecessary repeated environment checks."

Experience:
"For environment-version questions, use the active interpreter first and avoid redundant checks."
```

That experience can later be retrieved when solving similar tasks.

------------------------------------------------------------------------

# 27. Phase 5 Principle

> **Reflection should turn execution outcomes into actionable lessons,
> without yet changing or retrying the execution.**

This separation keeps the architecture modular:

``` text
Evaluate → Did it work?
Reflect  → Why / what should improve?
Memory   → What should we remember?
Retry    → How should we apply the improvement?
```

Each responsibility will be introduced in its own phase.
