# NanoCode --- Phase 7: Self-Improvement / Retry Loop

## 1. Goal

Phase 7 builds the first **self-improvement loop** on top of the Phase 6
memory system.

Up to Phase 6:

``` text
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
 ↓
Response
```

Phase 7 adds the ability to use evaluation and reflection to **improve a
failed execution and retry the task**.

Target architecture:

``` text
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
 ├── Success ───────────────→ Response
 │
 └── Failure
       ↓
    Reflect
       ↓
    Improve strategy
       ↓
    Retry
       ↓
    Evaluate again
       ↓
    Success or retry limit
```

> **Do not retry blindly. Every retry must be informed by evaluation,
> reflection, and available experience.**

------------------------------------------------------------------------

## 2. Phase 7 Scope

Implement:

-   Retry configuration
-   Evaluation-driven retry decisions
-   Reflection before retry
-   Improved execution context
-   Retry iteration tracking
-   Retry tracing
-   Maximum retry limit
-   Successful completion after a retry
-   Proper termination after repeated failure
-   Retry tests
-   End-to-end retry testing

Do **not** initially implement:

-   Infinite retries
-   Recursive agents
-   Multiple autonomous agents
-   Reinforcement learning
-   Model fine-tuning
-   Distributed execution
-   Long-running background agents
-   Automatic code commits
-   Automatic deployment

------------------------------------------------------------------------

## 3. Why Retry Comes After Memory

Phase 6 established:

``` text
Failure
 ↓
Reflection
 ↓
Experience
 ↓
Memory
```

Phase 7 closes the loop:

``` text
Failure
 ↓
Reflection
 ↓
Experience / Memory
 ↓
Improved strategy
 ↓
Retry
```

Therefore:

``` text
Phase 6 = Learn
Phase 7 = Act on what was learned
```

------------------------------------------------------------------------

## 4. Phase 7 Architecture

``` text
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
                              v
                          Evaluator
                              |
                    +---------+---------+
                    |                   |
                  SUCCESS             FAILURE
                    |                   |
                    v                   v
                 Response           Reflector
                                        |
                                        v
                                  Improvement
                                        |
                                        v
                                  Retry Planner
                                        |
                                        v
                                    Executor
                                        |
                                        v
                                   Evaluator
                                        |
                               +--------+--------+
                               |                 |
                             SUCCESS           FAILURE
                               |                 |
                               v                 v
                           Response       Retry limit reached
```

------------------------------------------------------------------------

## 5. Retry Configuration

Add retry settings to the existing `AgentConfig`.

Suggested:

``` python
@dataclass
class AgentConfig:
    plan_mode: bool = False
    max_retries: int = 2
```

Preserve all existing configuration fields.

`max_retries` means the number of retries **after the initial attempt**:

``` text
max_retries = 0
    → one execution only

max_retries = 1
    → initial attempt + one retry

max_retries = 2
    → initial attempt + two retries
```

------------------------------------------------------------------------

## 6. Retry State

Extend `AgentState` only if necessary.

A useful field is:

``` python
retry_count: int = 0
```

Prefer using:

``` text
state.config.max_retries
```

for the configured limit instead of duplicating configuration.

------------------------------------------------------------------------

## 7. Basic Retry Algorithm

The initial algorithm:

``` python
attempt = 0

while True:

    execute

    evaluation = evaluate

    if evaluation.success:
        break

    if attempt >= max_retries:
        break

    reflection = reflect

    improve strategy

    attempt += 1
```

The critical order is:

``` text
Execute
 ↓
Evaluate
 ↓
If failure:
    Reflect
    Improve
    Retry
```

------------------------------------------------------------------------

## 8. Retry Decision

The evaluator remains responsible for deciding whether execution
succeeded.

Use:

``` python
evaluation.success
```

Do not use:

``` python
if response:
```

or:

``` python
if len(response) > 0:
```

The flow is:

``` text
Executor
 ↓
Evaluator
 ↓
evaluation.success
```

------------------------------------------------------------------------

## 9. Reflection Before Retry

When:

``` text
evaluation.success == False
```

call:

``` python
reflection = self.reflector.reflect(
    state,
    evaluation,
)
```

The reflection should provide:

``` text
diagnosis
improvement
should_improve
```

Example:

``` text
Diagnosis:
The command used the wrong file path.

Improvement:
Inspect the project structure before retrying the command.
```

------------------------------------------------------------------------

## 10. Improving Retry Context

A retry should not simply repeat the original request unchanged.

Bad:

``` text
Attempt 1
 ↓
Failure
 ↓
Attempt 2
 ↓
same plan
 ↓
Failure
```

Better:

``` text
Attempt 1
 ↓
Failure
 ↓
Reflection
 ↓
Improvement
 ↓
Attempt 2 with improvement context
```

The planner should receive the previous failure information.

Example:

``` text
Previous attempt failed.

Diagnosis:
The command used the wrong file path.

Improvement:
Inspect the project structure before retrying.

Current task:
Fix the failing test.
```

------------------------------------------------------------------------

## 11. Memory During Retry

Phase 6 memory should remain available.

The retry planner can receive:

``` text
Current task
+
Relevant previous experiences
+
Current failure diagnosis
+
Current improvement suggestion
```

Conceptually:

``` text
                 Planner
                    ↑
       +------------+-------------+
       |            |             |
   Task context   Memory      Reflection
```

This combines longer-term experience with current-run feedback.

------------------------------------------------------------------------

## 12. Avoid Duplicate Memory

A single failed execution with several retries should not automatically
create many duplicate memories.

Avoid:

``` text
Experience 1
Experience 2
Experience 3
```

for the same failure.

Initially, prefer storing one actionable lesson for the retry cycle.

------------------------------------------------------------------------

## 13. Retry Count

Every retry should be visible.

``` text
attempt: 0
```

Initial execution.

``` text
attempt: 1
```

First retry.

``` text
attempt: 2
```

Second retry.

Keep executor iteration and agent retry count separate.

------------------------------------------------------------------------

## 14. Retry Tracing

Suggested trace events:

``` text
retry.started
retry.completed
retry.exhausted
```

Example:

``` text
[TRACE] ... | agent | retry.started
attempt: 1
reason: evaluation_failed
```

After success:

``` text
[TRACE] ... | agent | retry.completed
attempt: 1
status: success
```

When exhausted:

``` text
[TRACE] ... | agent | retry.exhausted
attempts: 3
```

Follow the existing `Tracer` API and naming conventions.

------------------------------------------------------------------------

## 15. Executor Iterations vs Agent Retries

These are different.

Executor iteration:

``` text
LLM
 ↓
tool call
 ↓
LLM
 ↓
tool call
 ↓
LLM
```

is one execution attempt.

Agent retry:

``` text
Attempt 1
 ↓
Evaluate
 ↓
Attempt 2
```

is a new attempt after failure.

Do not confuse:

``` text
executor.iteration
```

with:

``` text
agent.retry_count
```

------------------------------------------------------------------------

## 16. Retry Success

Example:

``` text
Attempt 0
 ↓
Evaluator: failure
 ↓
Reflector
 ↓
Retry
 ↓
Attempt 1
 ↓
Evaluator: success
 ↓
Return response
```

The final response should come from the successful attempt.

------------------------------------------------------------------------

## 17. Retry Exhaustion

For:

``` text
max_retries = 2
```

the maximum attempts are:

``` text
Attempt 0 → failure
Attempt 1 → failure
Attempt 2 → failure
```

Then:

``` text
retry.exhausted
```

No fourth attempt.

------------------------------------------------------------------------

## 18. Avoid Infinite Loops

Always enforce a hard retry limit.

Never implement:

``` python
while not success:
    retry()
```

without a termination condition.

------------------------------------------------------------------------

## 19. Suggested Run Structure

Conceptually:

``` python
def run(self, task: str) -> str:

    state = self.create_state(task)

    experiences = self.memory.retrieve(task)

    attempt = 0

    while True:

        self.planner.run(
            state,
            experiences=experiences,
        )

        self.executor.run(state)

        evaluation = self.evaluator.evaluate(state)

        if evaluation.success:
            break

        if attempt >= self.config.max_retries:
            break

        reflection = self.reflector.reflect(
            state,
            evaluation,
        )

        # Make reflection available to the next planning step.

        attempt += 1

    return state.final_response
```

This is architectural guidance, not code to copy blindly. Check the
existing implementations first.

------------------------------------------------------------------------

## 20. Retry Context

Keep retry context separate from the original task.

For example:

``` python
retry_context = {
    "attempt": attempt,
    "diagnosis": reflection.diagnosis,
    "improvement": reflection.improvement,
}
```

The planner can receive:

``` text
Current task
+
Relevant memory
+
Retry context
```

------------------------------------------------------------------------

## 21. Do Not Mutate the Original Task

Keep:

``` python
state.task
```

unchanged.

Do not turn:

``` text
Fix the failing test.
```

into:

``` text
Fix the failing test.
Previous failure: ...
Try something else.
```

Use separate retry context instead.

------------------------------------------------------------------------

## 22. Suggested Planner API

The planner may eventually accept:

``` python
planner.run(
    state,
    experiences=experiences,
    retry_context=retry_context,
)
```

Maintain backward compatibility:

``` python
planner.run(state)
```

should continue to work.

------------------------------------------------------------------------

## 23. Testing Strategy

Phase 7 needs tests for:

1.  No retry on success
2.  Retry after failure
3.  Retry context reaches planner
4.  Retry succeeds
5.  Retry fails repeatedly
6.  Retry limit is respected
7.  Retry count is correct
8.  Retry tracing
9.  Memory remains available during retry
10. Existing Phase 1--6 tests remain passing

------------------------------------------------------------------------

## 24. Test: Successful Task

Given:

``` text
evaluation.success = True
```

Expected:

``` text
executor called once
evaluator called once
retry count = 0
```

------------------------------------------------------------------------

## 25. Test: One Retry

Given:

``` text
Attempt 0 → failure
Attempt 1 → success
```

Expected:

``` text
executor called twice
evaluator called twice
reflection called once
retry count = 1
```

------------------------------------------------------------------------

## 26. Test: Retry Exhaustion

Given:

``` text
max_retries = 2
```

and:

``` text
Attempt 0 → failure
Attempt 1 → failure
Attempt 2 → failure
```

Expected:

``` text
executor called three times
evaluator called three times
agent stops
```

There must not be a fourth attempt.

------------------------------------------------------------------------

## 27. Test: Retry Context

After a failure:

``` text
diagnosis:
"Wrong command used."

improvement:
"Inspect the directory first."
```

The next planner call should receive both pieces of information.

This proves the retry actually changes planning context.

------------------------------------------------------------------------

## 28. Test: Memory + Retry

Seed memory with:

``` text
Previous task:
Fix Python test

Improvement:
Inspect traceback before modifying code.
```

Then trigger a retry.

The planner should receive both:

``` text
Relevant previous experience
```

and:

``` text
Current retry improvement
```

This verifies:

``` text
Memory + Reflection → Planner
```

------------------------------------------------------------------------

## 29. End-to-End Retry Test

After deterministic tests pass, create:

``` text
test/test_agent_retry_e2e.py
```

The E2E test should verify:

``` text
Real Planner
 ↓
Real Executor
 ↓
Real Evaluator
 ↓
Failure
 ↓
Real Reflector
 ↓
Retry
 ↓
Real Planner
 ↓
Real Executor
 ↓
Evaluator
```

Choose a task where retry is meaningful and bounded. Do not rely on an
arbitrary task randomly failing.

------------------------------------------------------------------------

## 30. Configuration Example

A user should eventually be able to configure:

``` python
config = AgentConfig(
    max_retries=2,
)

agent = NanoCodeAgent(
    config=config,
)
```

The agent must respect the configured limit.

------------------------------------------------------------------------

## 31. Retry Observability

A useful trace should look like:

``` text
planner.started
planner.completed

executor.started
executor.completed

evaluator.completed
success: False

reflector.completed
should_improve: True

retry.started
attempt: 1

planner.started
planner.completed

executor.started
executor.completed

evaluator.completed
success: True

retry.completed
attempt: 1
status: success
```

------------------------------------------------------------------------

## 32. Error Handling

If reflection itself fails:

``` text
Executor
 ↓
Evaluator → failure
 ↓
Reflector → error
```

the agent should not enter an uncontrolled retry loop.

A safe initial behavior is:

``` text
reflection failure
 ↓
stop retrying
 ↓
return current response
```

Follow existing project error-handling conventions.

------------------------------------------------------------------------

## 33. Retry and Final Response

The final response should come from the latest successful execution.

If all attempts fail, return the latest available response rather than
an empty string.

------------------------------------------------------------------------

## 34. Phase 7 Acceptance Criteria

Phase 7 is complete when:

-   [ ] `max_retries` configuration exists.
-   [ ] Retry count is tracked.
-   [ ] Successful executions do not retry.
-   [ ] Failed executions can retry.
-   [ ] Reflection occurs before retry.
-   [ ] Retry context reaches the planner.
-   [ ] Memory remains available during retry.
-   [ ] Successful retry terminates execution.
-   [ ] Retry exhaustion terminates execution.
-   [ ] Infinite retry is impossible.
-   [ ] Retry events are traced.
-   [ ] Retry count is observable.
-   [ ] Deterministic retry tests pass.
-   [ ] Memory + retry integration test passes.
-   [ ] Real LLM retry E2E test passes.
-   [ ] Existing Phase 1 tests pass.
-   [ ] Existing Phase 2 tests pass.
-   [ ] Existing Phase 3 tests pass.
-   [ ] Existing Phase 4 tests pass.
-   [ ] Existing Phase 5 tests pass.
-   [ ] Existing Phase 6 tests pass.

------------------------------------------------------------------------

## 35. Expected Project Structure

``` text
agent/
├── __init__.py
├── agent.py
├── state.py
├── config.py
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
├── test_planner_memory.py
├── test_agent_memory.py
├── test_agent_memory_integration.py
├── test_agent_memory_e2e.py
├── test_agent_retry.py
├── test_agent_retry_memory.py
└── test_agent_retry_e2e.py
```

------------------------------------------------------------------------

## 36. Phase 7 Architecture Checkpoint

After Phase 7:

``` text
                         NanoCodeAgent
                              |
                              v
                           Memory
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
                 +------------+------------+
                 |                         |
              SUCCESS                    FAILURE
                 |                         |
                 v                         v
              Response                  Reflector
                                           |
                                           v
                                      Improvement
                                           |
                                           v
                                      Retry Planner
                                           |
                                           v
                                        Executor
                                           |
                                           v
                                       Evaluator
                                           |
                              +------------+------------+
                              |                         |
                           SUCCESS                  FAILURE
                              |                         |
                              v                         v
                           Response               Retry limit
                                                     ↓
                                                   Stop
```

The agent now has the foundation of a feedback loop:

``` text
Execute
   ↓
Evaluate
   ↓
Reflect
   ↓
Improve
   ↓
Retry
```

------------------------------------------------------------------------

## 37. Phase 7 → Future Phases

Once the bounded retry loop is stable, later phases can explore:

``` text
Better codebase retrieval
        ↓
Better planning
        ↓
Better reflection
        ↓
Smarter retry selection
        ↓
Long-term persistent memory
```

Do not jump directly to autonomous recursive behavior.

Priority:

``` text
Correctness
 ↓
Observability
 ↓
Bounded behavior
 ↓
Reliability
 ↓
Advanced autonomy
```

------------------------------------------------------------------------

## 38. Phase 7 Principle

> **A retry is useful only when the agent has a reason to believe the
> next attempt will be better than the previous one.**

The core loop is:

``` text
Failure
   ↓
Understand why
   ↓
Change strategy
   ↓
Retry
   ↓
Evaluate again
```

A retry without reflection is just repetition.

A retry with reflection is the foundation of self-improvement.
