# NanoCode --- Phase 4: Evaluation

## Goal

Phase 4 adds an **evaluation layer** to NanoCode.

Up to Phase 3:

``` text
Task
 ↓
Plan
 ↓
Execute
 ↓
Trace
```

Phase 4 adds:

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

The evaluator answers:

> Did the agent successfully accomplish the task?

The initial implementation should remain simple and deterministic.

------------------------------------------------------------------------

## Why Evaluation Comes Before Reflection

The eventual self-improvement loop is:

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

Reflection needs an evaluation signal.

Therefore:

``` text
Phase 4 → Evaluation
Phase 5 → Reflection
Phase 6 → Experience / Memory
Phase 7 → Recursive Self-Improvement
```

Do not implement recursion or self-improvement in Phase 4.

------------------------------------------------------------------------

## Phase 4 Architecture

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
                 +------------+------------+
                 |            |            |
                 v            v            v
             SUCCESS       PARTIAL      FAILURE
```

The evaluator observes the completed execution. It does not control
execution yet.

------------------------------------------------------------------------

## EvaluationResult

Create a structured result:

``` python
@dataclass
class EvaluationResult:
    success: bool
    score: float
    reason: str
```

Example:

``` python
EvaluationResult(
    success=True,
    score=1.0,
    reason="Task completed successfully.",
)
```

Failure:

``` python
EvaluationResult(
    success=False,
    score=0.0,
    reason="Agent execution failed.",
)
```

The scoring policy can become more sophisticated later.

------------------------------------------------------------------------

## Initial Evaluation Rules

### 1. Completed execution

If:

``` python
state.status == AgentStatus.COMPLETED
```

and:

``` python
state.final_response.strip()
```

is non-empty, initially evaluate it as:

``` text
success = True
score = 1.0
```

### 2. Failed execution

If:

``` python
state.status == AgentStatus.FAILED
```

evaluate:

``` text
success = False
score = 0.0
```

### 3. Maximum iterations

If:

``` python
state.status == AgentStatus.MAX_ITERATIONS
```

evaluate:

``` text
success = False
score = 0.0
```

### 4. Empty final response

If execution reports completion but:

``` python
state.final_response.strip() == ""
```

evaluate:

``` text
success = False
score = 0.0
```

Adapt these status names to the exact `AgentStatus` implementation in
the current codebase.

------------------------------------------------------------------------

## `agent/evaluator.py`

Create:

``` text
agent/evaluator.py
```

Suggested structure:

``` python
from dataclasses import dataclass

from agent.state import AgentState, AgentStatus
from agent.tracer import Tracer


@dataclass
class EvaluationResult:
    success: bool
    score: float
    reason: str


class Evaluator:
    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer or Tracer()

    def evaluate(
        self,
        state: AgentState,
    ) -> EvaluationResult:

        if state.status == AgentStatus.COMPLETED:

            if state.final_response.strip():

                result = EvaluationResult(
                    success=True,
                    score=1.0,
                    reason="Task completed successfully.",
                )

            else:

                result = EvaluationResult(
                    success=False,
                    score=0.0,
                    reason="Agent completed without a final response.",
                )

        elif state.status == AgentStatus.FAILED:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason="Agent execution failed.",
            )

        elif state.status == AgentStatus.MAX_ITERATIONS:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason="Agent reached the maximum number of iterations.",
            )

        else:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason=f"Agent ended with status: {state.status}",
            )

        self.tracer.record(
            "evaluation.completed",
            component="evaluator",
            success=result.success,
            score=result.score,
            reason=result.reason,
        )

        return result
```

Adapt status names to the existing implementation rather than changing
`AgentStatus` just for this phase.

------------------------------------------------------------------------

## Evaluation Trace

A successful execution should eventually produce:

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
    success: True
    score: 1.0
```

A failed execution can produce:

``` text
executor.started
...
executor.failed

evaluation.completed
    success: False
    score: 0.0
```

The evaluator uses the same shared tracer.

------------------------------------------------------------------------

## Shared Tracer

Keep one tracer for a normal agent run:

``` text
                     NanoCodeAgent
                           |
                        Tracer
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
     Planner            Executor          Evaluator
        |                  |                  |
        +------------------+------------------+
                           |
                           v
                      events[]
```

Do not create unrelated tracers for each component.

------------------------------------------------------------------------

## Integrating Evaluation into NanoCodeAgent

After testing the evaluator independently, connect it to
`NanoCodeAgent`.

The intended flow is:

``` python
state = self.create_state(task)

self.planner.run(state)

self.executor.run(state)

evaluation = self.evaluator.evaluate(state)

return state.final_response
```

The evaluation result can later be stored on state if that becomes
necessary. Avoid adding unnecessary state fields at this stage.

------------------------------------------------------------------------

## Updated Agent Flow

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
AgentState.plan
 |
 v
Executor
 |
 +--> LLM
 |
 +--> Tools
 |
 v
Final Response
 |
 v
Evaluator
 |
 v
EvaluationResult
 |
 v
Return
```

The evaluator does not retry or modify the agent yet.

------------------------------------------------------------------------

## Testing Strategy

Create:

``` text
test/test_evaluator.py
```

The evaluator should first be tested without calling the LLM.

This makes the tests:

-   Fast
-   Deterministic
-   Cheap
-   Independent of API limits

### Test 1 --- Successful execution

Create an `AgentState` with:

``` text
status = COMPLETED
final_response = "Python 3.12.5"
```

Then:

``` python
result = evaluator.evaluate(state)

assert result.success is True
assert result.score == 1.0
```

### Test 2 --- Failed execution

Create:

``` text
status = FAILED
final_response = ""
```

Then:

``` python
result = evaluator.evaluate(state)

assert result.success is False
assert result.score == 0.0
```

### Test 3 --- Maximum iterations

Create:

``` text
status = MAX_ITERATIONS
```

Then:

``` python
result = evaluator.evaluate(state)

assert result.success is False
assert result.score == 0.0
```

### Test 4 --- Empty final response

Create:

``` text
status = COMPLETED
final_response = ""
```

Then:

``` python
result = evaluator.evaluate(state)

assert result.success is False
```

------------------------------------------------------------------------

## Important Limitation

This initial evaluator is an **execution-level evaluator**.

It does not yet know whether the actual coding task was correct.

For example:

``` text
Task:
"Fix the failing unit test."

Agent:
- Runs tests
- Makes an incorrect change
- Produces a final response
- Status = COMPLETED
```

The basic evaluator could incorrectly classify this as successful.

That limitation is intentional.

A stronger evaluator will eventually use:

-   Test results
-   Exit codes
-   Repository state
-   Task-specific verifiers
-   Terminal-Bench evaluation
-   External benchmark signals

Those capabilities should be added later.

------------------------------------------------------------------------

## Terminal-Bench Connection

Eventually:

``` text
NanoCode
   |
   v
Task execution
   |
   v
Terminal-Bench verifier
   |
   v
Benchmark result
```

That result can become:

``` python
EvaluationResult(
    success=True,
    score=1.0,
    reason="Terminal-Bench verifier passed.",
)
```

or:

``` python
EvaluationResult(
    success=False,
    score=0.0,
    reason="Terminal-Bench verifier failed.",
)
```

For Phase 4, first stabilize the evaluator abstraction. Terminal-Bench
integration comes after that.

------------------------------------------------------------------------

## Why This Matters for RLM

The future recursive system needs a reliable signal:

``` text
Attempt
   |
   v
Evaluate
   |
   +---- Success → Stop
   |
   +---- Failure → Reflect
```

Without evaluation:

``` text
Attempt
   |
   v
???
```

The agent has no reliable basis for deciding whether it should improve
and retry.

Phase 4 establishes the first decision boundary:

``` text
                    Evaluation
                        |
                 +------+------+
                 |             |
                 v             v
              SUCCESS       FAILURE
                 |             |
                 v             v
                DONE       Reflection
```

------------------------------------------------------------------------

## Phase 4 Acceptance Criteria

Phase 4 is complete when:

-   [ ] `EvaluationResult` exists.
-   [ ] `Evaluator` exists.
-   [ ] Evaluator accepts `AgentState`.
-   [ ] Successful completed executions can be evaluated.
-   [ ] Failed executions can be evaluated.
-   [ ] Maximum-iteration executions can be evaluated.
-   [ ] Empty final responses are rejected.
-   [ ] Evaluation events are recorded in the shared tracer.
-   [ ] `test/test_evaluator.py` passes.
-   [ ] Existing Phase 1 tests still pass.
-   [ ] Existing Phase 2 tests still pass.
-   [ ] Existing Phase 3 tests still pass.
-   [ ] Evaluation does not execute tools.
-   [ ] Evaluation does not retry tasks.
-   [ ] Evaluation does not perform reflection.
-   [ ] Evaluation does not implement recursion.

------------------------------------------------------------------------

## Phase 4 Deliverable

Expected structure:

``` text
agent/
├── __init__.py
├── agent.py
├── state.py
├── executor.py
├── planner.py
├── tracer.py
└── evaluator.py

test/
├── test_state.py
├── test_executor.py
├── test_planner.py
├── test_tracer.py
└── test_evaluator.py
```

The architecture becomes:

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
```

------------------------------------------------------------------------

## Phase 4 → Phase 5

Once Phase 4 is stable, Phase 5 adds **Reflection**.

Future flow:

``` text
Task
 |
 v
Plan
 |
 v
Execute
 |
 v
Evaluate
 |
 +---- SUCCESS
 |       |
 |       v
 |      DONE
 |
 +---- FAILURE
         |
         v
      Reflect
         |
         v
   Identify mistakes
```

Reflection will use:

``` text
AgentState
+
Trace
+
EvaluationResult
```

to answer:

> What went wrong, and what should the agent change on its next attempt?

That becomes the first step toward recursive self-improvement.

------------------------------------------------------------------------

## Phase 4 Principle

> **The agent must be able to distinguish between completing an
> execution and actually having a successful execution.**
