# NanoCode --- Phase 3: Structured Execution Tracing

## 1. Goal

Phase 3 upgrades NanoCode's basic tracer into a **structured execution
tracing system**.

The purpose is to record enough information about an agent run to
answer:

-   What did the agent do?
-   Which component performed the action?
-   How many LLM iterations occurred?
-   Which tools were used?
-   How long did operations take?
-   Where did errors occur?
-   What happened before and after a failure?

This trace will become the foundation for later:

-   Evaluation
-   Reflection
-   Experience / memory
-   Recursive self-improvement
-   Terminal-Bench analysis

------------------------------------------------------------------------

## 2. Current Architecture

Phase 1 and Phase 2 currently provide:

``` text
User
  |
  v
NanoCodeAgent.run(task)
  |
  v
AgentState
  |
  +--> Planner
  |      |
  |      +--> state.plan
  |
  v
Executor
  |
  +--> LLM
  |
  +--> Tools
  |
  +--> Update messages
  |
  v
Final Response
```

Basic trace events currently include:

``` text
planner.started
planner.completed
executor.started
llm.request
llm.response
tool.started
tool.completed
executor.completed
```

Phase 3 makes these events structured and useful for analysis.

------------------------------------------------------------------------

## 3. Phase 3 Architecture

``` text
                         NanoCodeAgent
                              |
                              v
                           Tracer
                              |
                 +------------+------------+
                 |                         |
                 v                         v
              Planner                  Executor
                 |                         |
                 |                         +--> LLM
                 |                         |
                 |                         +--> Tools
                 |                         |
                 +-------------+-----------+
                               |
                               v
                         tracer.events
```

There should be **one shared Tracer per agent instance**.

Planner and Executor write their events into the same trace.

------------------------------------------------------------------------

## 4. TraceEvent

The basic unit of tracing is `TraceEvent`.

``` python
@dataclass
class TraceEvent:
    name: str
    timestamp: str
    component: str
    data: dict[str, Any]
    duration_ms: float | None
    error: str | None
```

### Fields

#### `name`

The event name.

Examples:

``` text
planner.started
planner.completed
executor.started
llm.request
llm.response
tool.started
tool.completed
executor.completed
```

#### `timestamp`

The time at which the event was recorded.

#### `component`

The NanoCode component responsible for the event.

Examples:

``` text
agent
planner
executor
```

#### `data`

Additional event-specific information.

Examples:

``` python
{"task": "Fix failing tests"}
```

or:

``` python
{"tool": "bash"}
```

or:

``` python
{"iteration": 3, "tool_calls": 1}
```

#### `duration_ms`

Optional duration of the operation in milliseconds.

#### `error`

Optional error information.

------------------------------------------------------------------------

## 5. Tracer API

The tracer should expose:

``` python
tracer.record(...)
```

``` python
tracer.get_events()
```

``` python
tracer.clear()
```

The tracer should continue to support terminal output while also
retaining structured events in memory.

------------------------------------------------------------------------

## 6. `Tracer.record()`

Target interface:

``` python
def record(
    self,
    name: str,
    component: str = "agent",
    duration_ms: float | None = None,
    error: str | None = None,
    **data: Any,
) -> None:
    ...
```

Example:

``` python
tracer.record(
    "tool.completed",
    component="executor",
    duration_ms=142.5,
    tool="bash",
)
```

------------------------------------------------------------------------

## 7. Terminal Trace Output

Example:

``` text
[TRACE] 2026-08-09T00:30:10 | planner | planner.started
         task: Fix the failing tests

[TRACE] 2026-08-09T00:30:12 | planner | planner.completed
         steps: 5

[TRACE] 2026-08-09T00:30:12 | executor | executor.started
         task: Fix the failing tests

[TRACE] 2026-08-09T00:30:12 | executor | llm.request
         iteration: 1

[TRACE] 2026-08-09T00:30:14 | executor | llm.response
         iteration: 1
         finish_reason: tool_calls
         tool_calls: 1

[TRACE] 2026-08-09T00:30:14 | executor | tool.started
         tool: bash

[TRACE] 2026-08-09T00:30:14 | executor | tool.completed
         duration: 142.50 ms
         tool: bash

[TRACE] 2026-08-09T00:30:16 | executor | executor.completed
         iterations: 2
         status: completed
```

------------------------------------------------------------------------

## 8. Event Lifecycle

A normal agent run should produce a trace similar to:

``` text
planner.started
      |
      v
planner.completed
      |
      v
executor.started
      |
      v
llm.request
      |
      v
llm.response
      |
      +---- tool call ----+
      |                   |
      v                   v
tool.started       tool.completed
      |                   |
      +---------+---------+
                |
                v
           llm.request
                |
                v
           llm.response
                |
                v
       executor.completed
```

------------------------------------------------------------------------

## 9. Planner Tracing

The Planner should record at least:

### Start

``` python
self.tracer.record(
    "planner.started",
    component="planner",
    task=state.task,
)
```

### Completion

``` python
self.tracer.record(
    "planner.completed",
    component="planner",
    steps=len(state.plan),
)
```

Later, duration measurement will be added automatically.

------------------------------------------------------------------------

## 10. Executor Tracing

The Executor should record:

### Executor start

``` python
self.tracer.record(
    "executor.started",
    component="executor",
    task=state.task,
)
```

### LLM request

``` python
self.tracer.record(
    "llm.request",
    component="executor",
    iteration=state.iteration,
)
```

### LLM response

``` python
self.tracer.record(
    "llm.response",
    component="executor",
    iteration=state.iteration,
    finish_reason=finish_reason,
    tool_calls=len(tool_calls),
)
```

### Tool start

``` python
self.tracer.record(
    "tool.started",
    component="executor",
    tool=tool_call.name,
)
```

### Tool completion

``` python
self.tracer.record(
    "tool.completed",
    component="executor",
    tool=tool_call.name,
)
```

### Executor completion

``` python
self.tracer.record(
    "executor.completed",
    component="executor",
    iterations=state.iteration,
    status=state.status.value,
)
```

------------------------------------------------------------------------

## 11. Error Tracing

Errors should eventually become trace events rather than only terminal
exceptions.

Example:

``` python
try:
    ...
except Exception as e:
    self.tracer.record(
        "executor.failed",
        component="executor",
        error=str(e),
    )
    raise
```

The Planner should use the same pattern:

``` text
planner.started
planner.failed
```

This is important because future reflection needs to know **where and
why execution failed**.

------------------------------------------------------------------------

## 12. Duration Measurement

Phase 3 should measure operation duration automatically.

Conceptually:

``` text
start
  |
  v
LLM request
  |
  v
LLM response
  |
  v
end
```

Then:

``` text
duration_ms = end - start
```

The same approach can be used for:

-   Planner execution
-   LLM requests
-   Tool execution
-   Executor execution
-   Recursive attempts in later phases

The next implementation step is to add a timing mechanism so callers do
not have to calculate durations manually.

------------------------------------------------------------------------

## 13. Trace Data Model

A complete trace should eventually look like:

``` text
AgentRun
|
+-- run_id
+-- task
+-- start_time
+-- end_time
+-- status
|
+-- events[]
    |
    +-- TraceEvent
    |     +-- name
    |     +-- timestamp
    |     +-- component
    |     +-- duration_ms
    |     +-- data
    |     +-- error
    |
    +-- TraceEvent
    |
    +-- TraceEvent
    |
    +-- ...
```

A future serialized representation could be:

``` json
{
  "run_id": "run_001",
  "task": "Fix failing tests",
  "status": "completed",
  "events": [
    {
      "name": "planner.started",
      "component": "planner",
      "timestamp": "2026-08-09T00:30:10",
      "data": {
        "task": "Fix failing tests"
      },
      "duration_ms": null,
      "error": null
    }
  ]
}
```

The exact persistent format can be decided later.

------------------------------------------------------------------------

## 14. Testing

Phase 3 must preserve all previous tests.

Run:

``` powershell
python -m test.test_state
```

Run:

``` powershell
python -m test.test_planner
```

Run:

``` powershell
python -m test.test_executor
```

Run:

``` powershell
python -m test.test_tracer
```

The structured tracer test should verify:

-   Event count
-   Event name
-   Component
-   Event data
-   Duration
-   Error field

------------------------------------------------------------------------

## 15. Phase 3 Acceptance Criteria

Phase 3 is complete when:

-   [ ] `TraceEvent` contains structured metadata.
-   [ ] Every event has a timestamp.
-   [ ] Every event identifies its component.
-   [ ] Events can contain arbitrary metadata.
-   [ ] Events can contain duration.
-   [ ] Events can contain errors.
-   [ ] Planner produces trace events.
-   [ ] Executor produces trace events.
-   [ ] LLM requests/responses are traced.
-   [ ] Tool execution is traced.
-   [ ] The same Tracer is shared by Planner and Executor.
-   [ ] Existing Phase 1 tests still pass.
-   [ ] Existing Phase 2 tests still pass.
-   [ ] Tracer tests pass.

------------------------------------------------------------------------

## 16. Why Phase 3 Matters for Self-Improvement

The final goal is not simply debugging.

The trace will eventually become input to the self-improvement system.

The future architecture will be:

``` text
Task
 |
 v
Planner
 |
 v
Execution
 |
 v
Trace
 |
 v
Evaluation
 |
 v
Reflection
 |
 v
Experience
 |
 v
Improved Strategy
 |
 +--------------------+
 |                    |
 +----> Next Attempt -+
```

For example:

``` text
Attempt 1

Plan:
1. Run tests
2. Inspect failure
3. Fix code

Actions:
- bash: pytest
- file.read
- file.write

Result:
FAIL

Trace:
- pytest failed
- wrong file modified
- second test still failed

Reflection:
The agent modified the wrong configuration file.
```

Without structured traces, the reflection system has much less
information to reason about.

------------------------------------------------------------------------

## 17. Phase 3 → Phase 4

After structured tracing is complete, the next phase is:

**Phase 4 --- Evaluation**

The architecture becomes:

``` text
                 Agent
                   |
                   v
                Execute
                   |
                   v
                 Trace
                   |
                   v
               Evaluator
                   |
             +-----+-----+
             |           |
             v           v
          SUCCESS      FAILURE
```

The evaluator will determine whether the task was actually solved.

That evaluation signal will eventually enable:

``` text
failure
   |
   v
reflection
   |
   v
improved strategy
   |
   v
retry
```

------------------------------------------------------------------------

## 18. Phase 3 Deliverable

At the end of Phase 3:

``` text
agent/
├── __init__.py
├── agent.py
├── state.py
├── executor.py
├── planner.py
└── tracer.py

test/
├── test_state.py
├── test_executor.py
├── test_planner.py
└── test_tracer.py
```

The core execution architecture remains:

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
Structured Execution History
```

### Phase 3 principle

> **The agent should not merely execute a task; it should produce a
> structured record of how it attempted to execute the task.**
