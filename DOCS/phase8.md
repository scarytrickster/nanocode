# Phase 8 — Interactive CLI Coding Agent

## Goal

Turn NanoCode into a usable interactive terminal coding agent with a Claude-Code-like workflow.

The user should be able to start NanoCode from the terminal, see a startup banner, interact continuously, use slash commands/modes, and exit with `Ctrl+C`.

## Principles

- Keep the existing `NanoCodeAgent` architecture intact.
- Keep CLI/UI code separate from agent orchestration.
- Do not introduce LangChain or LangGraph unless a concrete need appears.
- Prefer lightweight terminal libraries such as Rich and prompt_toolkit where useful.
- Keep all existing tests passing.
- Do not put the interactive loop inside `agent/agent.py`.

## 8.1 — CLI Package and Entry Point

Create a dedicated CLI layer.

Suggested structure:

```text
nanocode1/
├── agent/
├── cli/
│   ├── __init__.py
│   ├── app.py
│   ├── banner.py
│   ├── commands.py
│   ├── prompt.py
│   └── renderer.py
├── config/
├── models/
└── test/
```

Requirements:
- Create a CLI entry point.
- CLI initializes `NanoCodeAgent`.
- CLI detects the current working directory.
- CLI does not duplicate agent logic.

Acceptance:
- NanoCode starts from the terminal.
- Existing agent tests remain unaffected.
- CLI can start without executing a task immediately.

## 8.2 — Interactive REPL

NanoCode should remain active after startup.

```text
nanocode >

> explain this project

... agent response ...

nanocode >

> find the failing tests

... agent response ...

nanocode >
```

Requirements:
- Interactive input loop.
- Each normal user message is passed to `agent.run()`.
- Display the final response.
- Preserve the session until exit.
- Handle empty input gracefully.

## 8.3 — Ctrl+C / Exit Handling

Requirements:
- `Ctrl+C` terminates the interactive session cleanly.
- No traceback for normal interruption.
- Display a short goodbye message.
- Handle EOF gracefully where practical.

Expected:

```text
nanocode > ^C

Goodbye!
```

## 8.4 — Startup Banner

Display useful runtime information, for example:

```text
╭──────────────────────────────────────────────╮
│                  NANOCODE                    │
│                                              │
│  Model: openrouter/free                      │
│  Workspace: D:\projects\myapp                │
│  Memory: enabled                             │
│  Tools: bash • files • web                   │
╰──────────────────────────────────────────────╯
```

Include where available:
- NanoCode name/version
- Active model
- Workspace
- Memory status
- Available tools
- Current mode

Consider `rich` for formatting.

## 8.5 — Workspace Detection

Starting from:

```powershell
cd D:\projects\calculator
nanocode
```

should show:

```text
Workspace: D:\projects\calculator
```

Requirements:
- Detect current working directory.
- Expose workspace to the CLI.
- Ensure tools operate relative to the intended workspace.
- Avoid accidentally operating on the NanoCode source repository when NanoCode is being used elsewhere.

Future support:

```powershell
nanocode D:\projects\calculator
```

Current-working-directory support is sufficient for the first implementation.

## 8.6 — Command System

Support commands separate from normal agent prompts:

```text
/help
/model
/status
/memory
/clear
/plan
/code
/chat
/exit
```

Initial behavior:
- `/help` — show commands.
- `/model` — show configured model.
- `/status` — show workspace, model, mode, memory, retry configuration.
- `/memory` — show basic memory information.
- `/clear` — clear current conversational/session state where supported.
- `/exit` — exit CLI.

Slash commands belong to the CLI layer, not the LLM.

## 8.7 — Session / Conversation State

The session should track:

```text
current workspace
current mode
conversation history
agent configuration
```

Keep session state separate from long-term agent memory.

Do not automatically turn every conversation message into a long-term memory experience.

## 8.8 — Agent Output Rendering

Make agent activity readable:

```text
nanocode > Fix the failing tests

● Planning...
✓ Plan ready

● Executing...
  → bash: pytest

● Evaluating...
✓ Tests passed

────────────────────────────────
Fixed the failing test in ...
────────────────────────────────
```

Use existing trace events where practical. Do not rewrite the tracing system just for UI presentation.

## 8.9 — Interactive Modes

Initial modes:

```text
/chat
/code
/plan
```

- Chat mode: general conversation.
- Code mode: repository/code changes.
- Plan mode: focus on understanding and planning before changes.

Requirements:
- Display current mode.
- Switch modes without restarting.
- Pass mode information to the agent only where useful.
- Do not create unnecessary separate agents.

## 8.10 — CLI Testing

Test without real LLM calls.

Cover:
- CLI initialization
- banner generation
- workspace detection
- command parsing
- `/help`
- `/model`
- `/status`
- `/memory`
- `/clear`
- `/exit`
- mode switching
- empty input
- Ctrl+C handling
- multiple tasks in one session

Use fake/mock agents.

Do not make API calls from the normal CLI test suite.

## 8.11 — Real Project Testing

After deterministic tests pass, use a disposable project:

```text
calculator/
├── calculator.py
└── test_calculator.py
```

Example tasks:

```text
Find the failing tests and fix them.

Add a divide() function and tests.

Inspect this project and explain its structure.

Find obvious bugs and fix them.
```

Observe:
- Planner
- Tool usage
- Memory retrieval
- Evaluator
- Reflector
- Retry
- File modifications
- Test execution

Do not initially test on the NanoCode source repository.

## 8.12 — Framework Evaluation

Do not add LangChain/LangGraph automatically.

### LangGraph

Consider later for:
- complex state graphs
- branching workflows
- parallel agent tasks
- human approval nodes
- sub-agents
- complex recovery workflows

NanoCode already implements planner/executor/evaluator/reflector/retry orchestration, so replacing it during Phase 8 is not required.

### LangChain

Consider later for:
- model abstraction
- tool abstraction
- structured output
- integrations

Do not add it merely for abstraction if the current implementation is simpler.

### Other Useful Libraries

Potential candidates:
- `rich` — terminal UI/rendering
- `prompt_toolkit` — interactive terminal input
- `typer` — CLI parsing
- `pydantic` — structured configuration/state

Add dependencies only when they solve an actual problem.

## Phase 8 Completion Criteria

- [ ] `nanocode` launches from the terminal.
- [ ] Startup banner displays runtime information.
- [ ] Current workspace is detected.
- [ ] Interactive REPL stays open after each task.
- [ ] Multiple tasks work in one session.
- [ ] `Ctrl+C` exits cleanly.
- [ ] `/help` works.
- [ ] `/model` works.
- [ ] `/status` works.
- [ ] `/memory` works.
- [ ] `/clear` works.
- [ ] `/exit` works.
- [ ] Modes can be switched.
- [ ] Agent output is readable.
- [ ] CLI has deterministic tests.
- [ ] Existing Phase 1–7 tests pass.
- [ ] NanoCode works on a separate test project.
- [ ] Phase 8 changes are committed and pushed.

## Recommended Order

```text
8.1 CLI package
      ↓
8.2 Interactive REPL
      ↓
8.3 Ctrl+C
      ↓
8.4 Banner
      ↓
8.5 Workspace
      ↓
8.6 Commands
      ↓
8.7 Session state
      ↓
8.8 Output rendering
      ↓
8.9 Modes
      ↓
8.10 CLI tests
      ↓
8.11 Real project test
      ↓
8.12 Framework evaluation
      ↓
Phase 8 complete
```

## Suggested Git Checkpoints

```text
feat: add interactive cli
feat: add nanocode startup banner
feat: add workspace detection
feat: add cli commands
feat: add interactive modes
test: add cli tests
feat: improve terminal rendering
```
