# Current Agent Flow

This document describes the current `agent/` package flow after the class-based refactor.

## High-Level Flow

```text
User
  |
  v
NanoCodeAgent.run(task)
  |
  v
NanoCodeAgent.create_state(task)
  |
  v
AgentState
  |
  v
Executor.run(state)
  |
  v
Call LLM -> Parse Tool Calls -> Execute Tools -> Update Messages
  |
  v
state.final_response
  |
  v
return final_response
```

## Main Components

### `NanoCodeAgent`

Defined in `agent/agent.py`.

`NanoCodeAgent` is the high-level facade used to run a task. It owns:

- `tools`
- `config`
- `executor`
- `messages`

Its main methods are:

- `create_state(task)`: appends the user task to conversation messages and creates an `AgentState`.
- `run(task)`: creates state, passes it to the executor, and returns `state.final_response`.

## `AgentState`

Defined in `agent/agent.py`.

`AgentState` holds the mutable runtime data for one agent run:

- `task`
- `messages`
- `tools`
- `config`
- `iteration`
- `status`
- `final_response`
- `tools_by_name`
- `tool_schemas`

`tools_by_name` and `tool_schemas` are built automatically in `__post_init__`.

## `Executor`

Defined in `agent/agent.py`.

`Executor.run(state)` owns the main execution loop:

1. Increment the iteration counter.
2. Call the configured LLM through `client.chat.completions.create(...)`.
3. Parse the streamed response using `parse_tool_calls(...)`.
4. If tool calls are present:
   - append the assistant tool-call message
   - execute tools through `execute_tool_calls(...)`
   - continue the loop
5. If no tool calls are present:
   - append the assistant final message
   - set `state.status = "complete"`
   - set `state.final_response`
   - return state

If the API call fails, the executor sets:

```text
state.status = "error"
state.final_response = "Error: ..."
```

If the maximum iteration count is reached, it sets:

```text
state.status = "max_iterations"
```

## Supporting Files

### `agent/executor.py`

Contains helper functions used by `Executor`:

- `parse_tool_calls(stream)`: reads streamed LLM chunks, prints assistant text, and collects tool calls.
- `execute_tool_calls(...)`: parses tool arguments, checks plan mode and approval rules, executes tools, and appends tool results to messages.

### `agent/state.py`

Contains:

- `get_system_prompt()`

This builds the initial system prompt using:

- current working directory
- OS information
- Python version
- files in the current directory
- optional `NANOCODE.md` project instructions

### `agent/__init__.py`

Exports the public package API:

- `AgentState`
- `Executor`
- `NanoCodeAgent`
- `main`
- `run_agent`
- `get_system_prompt`

## CLI Flow

The CLI entry point is still `main()` in `agent/agent.py`.

```text
main()
  |
  v
print_banner()
  |
  v
agent = NanoCodeAgent()
  |
  v
read user input in a loop
  |
  v
agent.config = AgentConfig(plan_mode=plan_mode)
  |
  v
agent.run(user_input)
```

The `/plan` command toggles plan mode. In plan mode, write tools are blocked by `execute_tool_calls(...)`.

## Compatibility API

`run_agent(messages, tools, config)` still exists for older code.

It creates an `AgentState` directly and passes it to `Executor.run(state)`.

This keeps existing imports working while the newer flow uses `NanoCodeAgent`.

Phase 1 :

User
│
▼
NanoCodeAgent.run(task)
│
▼
create_state(task)
│
▼
AgentState
│
▼
Executor.run(state)
│
├── Call LLM
├── Parse Tool Calls
├── Execute Tools
├── Update Messages
└── Repeat
│
▼
state.final_response
│
▼
Return Response
