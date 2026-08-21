# NanoCode

NanoCode is an AI coding agent built to explore a specific engineering question: what does it actually take to make an LLM agent operate safely and predictably on a real repository? It plans before it acts, interacts with the codebase exclusively through tools, evaluates its own output, and retries with information derived from why the previous attempt failed. On top of that single-agent pipeline sits a **Recursive Language Model (RLM)** path that decomposes a complex task into independent child investigations, runs them as bounded-concurrency child agents, and synthesizes their findings into one evidence-weighted answer.

Most of the engineering here is defensive rather than generative. An agent that can search a repository will happily read a 6,000-file `.venv`; an agent that calls tools in a loop will accumulate a context window far larger than the model accepts; a provider that rate-limits will do so in the middle of a fan-out. NanoCode addresses each of these with an explicit, tested layer: repository-exploration filtering, per-tool output limits, a context budget with deterministic compression, bounded child concurrency, child failure isolation, and application-level rate-limit classification. Every one of those layers is covered by deterministic tests that run without a network connection.

**Status: a working research/engineering project, not production software.** The full deterministic test suite passes offline, and the architecture is stable across its component contracts. What has *not* happened yet is comprehensive validation against a live model, or any standardized benchmark run — no Terminal-Bench score exists, and none is claimed. See [Known Limitations](#known-limitations) and [Terminal-Bench / Evaluation](#terminal-bench--evaluation).

---

## Key Features

| # | Capability | Status | What it does |
|---|---|---|---|
| 1 | Planner / Executor architecture | Implemented | `Planner` produces a step plan; `Executor` runs the tool-calling loop against it. Separate LLM calls, separate responsibilities. |
| 2 | Tool-based coding workflow | Implemented | Nine tools: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `task`, `todo_write`, `web_fetch`, `web_search`. |
| 3 | Repository exploration filtering | Implemented | `tools/ignore.py` prunes dependency/build/cache directories during recursive traversal; explicit paths still work. |
| 4 | Tool-output limiting | Implemented | Every large tool result is capped at 20,000 characters with an explicit truncation notice. |
| 5 | RLM routing | Implemented | Deterministic keyword/weight scoring decides `normal` vs `rlm` per task. No LLM call. |
| 6 | Deterministic RLM decomposition | Implemented (default) | Strategy-based decomposition into 1–3 complementary child investigations. |
| 7 | LLM-based RLM decomposition | Implemented (opt-in) | Task-specific child investigations from the model, strictly validated, with deterministic fallback. |
| 8 | Parallel RLM child execution | Implemented | Bounded thread pool (default 3 concurrent children); results returned in decomposition order. |
| 9 | RLM failure isolation | Implemented | A child that crashes becomes a failed result; its siblings run to completion. |
| 10 | Rate-limit handling | Implemented | SDK retries disabled, `RateLimitError` classified explicitly, bounded per-child retry, partial results reported honestly. |
| 11 | RLM CLI observability | Implemented | Child agent events are forwarded into the parent tracer and rendered with child attribution. |
| 12 | RSI retry/improvement loop | Implemented | A failed attempt's evaluation and reflection are fed into the next attempt's planning context. |
| 13 | Context budgeting | Implemented | Every LLM request is measured against a configured token budget before it is sent. |
| 14 | Deterministic context compression | Implemented | Old tool/assistant content is compressed head-and-tail; no second model is involved. |
| 15 | Benchmark / metrics infrastructure | Implemented (offline) | Scripted-model benchmark comparing the normal and RLM paths, plus context-budget scenarios. |

---

## Architecture

### Normal path

```
User Task
   |
   v
RLM Router  ---> (strategy = normal)
   |
   v
Planner  ------------------> LLM
   |
   v
Executor  -----------------> LLM (tool-calling loop)
   |
   v
Tools (bash / grep / read_file / edit_file / ...)
   |
   v
Evaluator
   |
   v
Reflector
   |
   v
RSI / Retry  --(improved context)--> back to Planner
   |
   v
Final Answer
```

### RLM path

```
User Task
   |
   v
RLM Router  ---> (strategy = rlm)
   |
   v
RLM Orchestrator
   |
   v
Decomposer  (deterministic by default, LLM opt-in)
   |
   v
RLM Runtime  (budget reservation, then bounded parallel dispatch)
   |
   +---------------+---------------+---------------+
   |               |               |               |
   v               v               v               |
Child Agent 1  Child Agent 2  Child Agent 3        |  (max 3 concurrent
   |               |               |               |   by default)
   +---------------+---------------+---------------+
                   |
                   v
        Evidence-based Synthesis
                   |
                   v
             Final Answer
```

Children execute **concurrently with bounded concurrency** (`max_concurrency`, default 3), but their results are collected into pre-allocated slots by decomposition index, so **synthesis always receives them in decomposition order** regardless of which finished first. Child 2 remains "child 2" in every trace event even when it completes before child 1.

Each child is an independent `NanoCodeAgent` with its own context budget, its own tool-output limits, and its own RSI retry loop. Child failures are isolated: an exception inside one child becomes a failed `RLMResult` in that child's slot and never cancels a sibling. Children are created with `rlm_enabled=False`, so a child can never open a second RLM hierarchy.

---

## RLM Architecture

| Component | File | Responsibility |
|---|---|---|
| `RLMRouter` | `rlm/router.py` | Decides `normal` vs `rlm` from deterministic phrase weights (threshold 0.5). It decides *whether*, never *what*. |
| `RLMOrchestrator` | `rlm/orchestrator.py` | Joins the routing decision to the RLM machinery: decompose, dispatch, synthesize, render the answer. Owns no execution logic itself. |
| `RLMRuntime` | `rlm/runtime.py` | The only component that executes children. Reserves budget, dispatches on a bounded pool, collects ordered results, wraps each child in a Langfuse span. |
| `RLMContext` | `rlm/context.py` | One child's task, content, depth and metadata. `child()` derives an isolated context with copied metadata at `depth + 1`. |
| `RLMDecomposer` | `rlm/decomposer.py` | The abstract interface: `decompose(task, context) -> list[RLMChildTask]`. Returns descriptions; never executes. |
| `DeterministicRLMDecomposer` | `rlm/decomposer.py` | **Default.** Matches the task against a small ordered strategy list (investigation / comparison / survey), falling back to a single child. No LLM, no network. |
| `LLMRLMDecomposer` | `rlm/llm_decomposer.py` | **Opt-in.** One tool-free model call returning strict JSON, then validation, deduplication and capping. |
| `RLMChildTask` | `rlm/decomposer.py` | A `NamedTuple` of `(task, content)` — the child's specialized investigation plus the original user task. |
| `NanoCodeCallHandler` | `rlm/nanocode_handler.py` | Adapts an `RLMContext` to a fresh `NanoCodeAgent` run; owns rate-limit classification, bounded retry, and child event forwarding. |
| `RLMSynthesizer` | `rlm/synthesizer.py` | Combines child results into findings with status (`confirmed` / `likely` / `possible` / `conflicting` / `unsupported`), confidence, and a primary finding. Deterministic — no LLM call. |
| `RLMBudget` | `rlm/budget.py` | The authority on execution limits: `max_depth=3`, `max_children=4`, `max_iterations=10` by default. |

**Decomposition strategy is explicit.** `DEFAULT_DECOMPOSITION_STRATEGY` is `"deterministic"`; LLM decomposition is enabled with `RLMOrchestrator(decomposition_strategy="llm")`. It is opt-in on purpose: enabling it by default would add a model call to every RLM run — on a provider that has already demonstrated rate-limit sensitivity — and would make the offline test suite and benchmark depend on a network call.

**Fallback behavior.** The LLM decomposer treats model output as untrusted. If the call raises (rate limit, connection error, timeout, anything), or the response is unparseable, missing `children`, not a list, empty, all-invalid, or all-duplicates, it falls back to `DeterministicRLMDecomposer`. There is exactly one attempt and no retry loop, and the fallback makes no second model call — a failed decomposition costs one request. The fallback is never silent: it is recorded in `last_outcome` and emitted as a `rlm.decomposition.fallback` trace event.

---

## Recursive Self-Improvement (RSI)

```
Attempt N
   |
   v
Evaluator      --> success? --> yes --> done
   |
   | no
   v
Reflector      --> diagnosis + improvement
   |
   v
RSIContext     --> attempt number, previous evaluation,
   |                previous reflection, previous response
   v
Improved planning context
   |
   v
Attempt N+1
```

`RSIContext` (`agent/rsi.py`) is a frozen snapshot carrying the attempt number, the previous `EvaluationResult`, the previous `ReflectionResult`, and a truncated previous response (500 characters). Its `to_planner_context()` renders into the shape `Planner.run(retry_context=...)` already accepted, so no pipeline stage changed shape.

RSI **extends the existing retry loop rather than adding a second one.** It reads and advances `state.retry_count` against `AgentConfig.max_retries` (default 2), so the number of attempts is exactly `max_retries + 1` — RSI has no counter of its own and cannot bypass the limit. Human rejection returns before evaluation, so it produces no reflection, no retry and no RSI events.

The key property is that attempt N+1 provably differs from attempt N: the attempt number and the evaluation reason are always present in a retry's planning context, so the prompt changes even when reflection produced nothing usable. Nothing is invented to fill that gap — an empty reflection is reported as `has_improvement=False`.

---

## Context Engineering

Three independent layers, each solving a different problem:

```
Repository filtering        (what gets discovered)
        |
        v
Tool output limiting        (how big one result can be)
        |
        v
Context budget management   (how big the whole request can be)
        |
        v
              LLM request
```

### 1. Repository filtering — `tools/ignore.py`

Recursive exploration prunes directories that are not the user's source:

```
.venv   venv   env   __pycache__   .git   node_modules
dist    build  .pytest_cache  .mypy_cache  .ruff_cache  .tox
```

Filtering applies to *traversal below a search root*, not to access. `read_file` on a path inside `.venv` still works, and `grep path=".venv"` still searches it — only `grep path="."` skips it.

### 2. Tool-output limiting — `tools/output_limit.py`

`DEFAULT_MAX_OUTPUT_CHARS = 20_000`. Output above the limit keeps its beginning and gains an explicit notice, so the agent knows it did not see everything:

```
[OUTPUT TRUNCATED]
Original: 499,178 characters
Returned: 20,000 characters
```

### 3. Context budget — `agent/context_budget.py`

`ContextBudgetManager` runs before every LLM request in both `Planner` and `Executor`. It:

- **estimates** token usage (`estimate_tokens`, characters ÷ 4 — named as an estimate, not a count);
- **preserves** all system messages, the original user task, and the most recent exchanges (`MIN_RECENT_MESSAGES`);
- **keeps assistant `tool_calls` and their answering `tool` messages together** as one indivisible unit, so a compressed conversation is still a valid API request;
- **compresses old tool content first**, then old assistant content, keeping head and tail of each entry;
- **drops whole old exchanges only when compression is insufficient**, replacing them with a counts-only `[CONTEXT COMPRESSED]` digest;
- **raises `ContextBudgetError` locally** — before any network request — if the protected core still cannot fit, stating the budget, the required size, and what could not be preserved.

Compression is deterministic and offline: **no second model is called to summarize context.** Doing so would add cost, another rate-limit surface, and a context problem inside the fix for the first one. Roles are never rewritten, so tool output cannot become a system instruction.

---

## Important Context-Budget Result

Reproducing the shape of the original context explosion — 80 exchanges whose tool results were *already* capped at the 20,000-character limit:

| Measurement | Value |
|---|---|
| Original | ~400,483 estimated tokens |
| After compression | ~30,921 estimated tokens |
| Configured budget | 170,393 estimated tokens |
| Model context limit | 262,144 tokens |
| Compression passes | 1 |
| Entries compressed | 77 |
| Entries dropped | 0 |
| Hard cap triggered | false |
| Roles preserved | `system`, `user`, `assistant`, `tool` |
| Tool-call pairing | preserved |
| Network calls | 0 |

This is a **synthetic deterministic regression** reproducing the earlier failure shape (a real run once reached 371,405 input tokens against a 262,144-token limit). It is a measurement of the context layer, **not a live model benchmark** — no request was made, and it says nothing about answer quality.

---

## Rate-Limit Handling

### The problem encountered

A real RLM run produced HTTP 429s from OpenRouter:

```
Error code: 429 — Provider returned error
poolside/laguna-s-2.1:free is temporarily rate-limited upstream
provider_name: Poolside
limit_source: upstream_provider_shared_pool
```

Investigation established that children run sequentially (concurrency was not the cause), that both 429s hit during **child planning** — the first LLM request each child makes — and that the OpenAI SDK's default `max_retries=2` silently turned one application-level call into **three HTTP attempts**, measured at ~2.4s per failure. Retrying was amplifying requests into a shared pool that was already saturated.

### The architectural response

- **SDK retries disabled** (`SDK_MAX_RETRIES = 0`) — one clear retry owner instead of two, one of them invisible.
- **Explicit classification** — only a real `openai.RateLimitError` is classified `rate_limit`; connection errors, timeouts and ordinary exceptions are not, and are never retried.
- **Bounded per-child retry** — one retry by default, inside the child's existing execution slot, so a retry never opens a new concurrency slot and never scales with child count.
- **Partial-result reporting** — a run where 1 of 3 children succeeded says so, in the metadata and in the user-facing answer, rather than presenting a third of an investigation as complete.
- **Failure isolation** — a rate-limited child does not cancel its siblings.

**This does not eliminate 429s.** The limit is upstream and provider-side; a free shared pool will keep rate-limiting. What changed is that failures are now bounded, correctly attributed, and honestly reported. Bounded concurrency (default 3) also means RLM can issue up to three simultaneous requests, which genuinely raises burst probability — that trade-off is measured and reported by the benchmark rather than hidden.

---

## Observability

A single `Tracer` (`agent/tracer.py`) records structured `TraceEvent`s and dispatches them to an optional callback. The CLI passes `renderer.handle_trace` as that callback. RLM child agents forward their **real** events into the parent tracer — nothing is simulated, and there is no separate print-based event system. The tracer serializes delivery with a lock so interleaved events from parallel children stay readable and correctly attributed.

Example CLI output during an RLM run:

```
● RLM: analyzing from multiple perspectives...
● RLM: decomposing task (deterministic)...
● RLM: decomposed into 3 child tasks
● RLM child 1/3
● [RLM child 1/3] Planning...
✓ [RLM child 1/3] Plan ready
● [RLM child 1/3] Executing...
  ↳ [RLM child 1/3] read_file
  ✓ [RLM child 1/3] read_file
✓ [RLM child 1/3] Execution complete
✓ RLM child 1 complete
  ↳ context compressed: 400k -> 30k tokens
✓ RLM: synthesis complete
```

Event families: `routing.decided`, `rlm.started`, `rlm.decomposition.*`, `rlm.child.*`, `rlm.synthesis.completed`, `planner.*`, `llm.*`, `tool.*`, `evaluation.completed`, `reflection.completed`, `retry.*`, `rsi.*`, `memory.*`, `context.compressed`, `context.budget.exceeded`. Trace metadata carries counts, sizes and flags only — never prompts, tool output, model responses, or credentials.

**Langfuse** integration exists in the codebase: `agent/agent.py`, `planner.py`, `executor.py`, `evaluator.py`, `reflector.py` and `rlm/runtime.py` open spans (`nanocode-run`, `routing-decision`, `rlm-decomposition`, `rlm-child`, `rlm-synthesis`), and child spans propagate into worker threads via `contextvars.copy_context()`. Span structure is verified **offline** with a recording stand-in client. There is a `tests/test_langfuse.py`, but it requires live credentials and is **not** part of the deterministic suite — live Langfuse validation has not been performed.

---

## Safety and Reliability Engineering

| Protection | Why it exists |
|---|---|
| Bounded tool output (20k chars) | One `find . -type f` produced ~499,000 characters — ~125k tokens from a single tool call. |
| Ignored dependency/build/cache directories | Recursive exploration walked 6,652 files / 53.2 MB, almost all of it vendored code. |
| Bounded RLM concurrency (default 3) | Unbounded fan-out would turn parallelism into a request storm against a rate-limited provider. |
| RLM child failure isolation | One crashing investigation must not destroy the two that succeeded. |
| Bounded rate-limit retry | Unbounded retry against a saturated shared pool makes the problem worse, not better. |
| Context budget | Bounded individual outputs still sum to an unbounded conversation. |
| Deterministic compression | An LLM-based summarizer would add cost, latency, and a second context problem. |
| Protected system/user roles | Compression must never let tool output become a system instruction (prompt-injection surface). |
| Tool-call pairing preservation | Dropping a `tool` message while keeping its `assistant` `tool_calls` produces an invalid request. |
| No tools for the LLM decomposer | Decomposition decides *what* to investigate; only child agents decide *how*. |
| Decomposition fallback | A model returning malformed JSON must degrade to safe deterministic children, not fail the run. |
| Explicit partial-result reporting | A one-of-three investigation reported as a clean success is worse than a reported failure. |

---

## Testing

**Latest verified result: 554 deterministic tests passed** (plus `test_agent_contract.py`: 14 passed, 0 failed), run offline with no OpenRouter access. Rerun locally to confirm — the number reflects the state at the end of the context-budget work.

Representative suites:

| Suite | Focus |
|---|---|
| `tests/test_context_budget.py` | Token estimation, compression, hard cap, 371k regression, role preservation |
| `tests/test_rlm_parallel.py` | Concurrency proven with barriers, ordering, budgets, thread cleanup |
| `tests/test_rlm_llm_decomposition.py` | Schema validation, deduplication, fallback, injection safety |
| `tests/test_rlm_synthesis_evidence.py` | Finding grouping, agreement, contradiction, primary selection |
| `tests/test_agent_rsi.py` | Improvement-context propagation, retry limits, memory correctness |
| `tests/test_rlm_rate_limit.py` | 429 classification, bounded retry, partial results |
| `tests/test_rlm_cli_events.py` | Child event forwarding and renderer attribution |
| `tests/test_rlm_end_to_end.py` | Full router → orchestrator → runtime → synthesis path |
| `tests/test_rlm_decomposition.py` | Deterministic decomposition contracts |
| `tests/test_rlm_router.py` | Routing decisions and signal weighting |
| `tests/test_rlm_contract.py` | Core RLM type/contract invariants |
| `tests/test_tool_exploration_filtering.py` | Directory pruning and explicit-access behavior |
| `tests/test_tool_output_limit.py` | Per-tool truncation and notices |
| `tests/test_agent_identity.py` | System/user boundary and identity protection |

Testing discipline applied throughout:

- No tests were deleted, skipped, or weakened to make an implementation pass.
- No budget or limit was raised to make a test pass.
- Deterministic tests require no OpenRouter key and make no network calls; several install tripwires that fail if a request is attempted.
- Concurrency is proven with `threading.Barrier`/`Event` and a negative control, never with `sleep` or wall-clock timing.
- **Live LLM validation has not been completed.** Suites requiring a real model (`test_langfuse.py`, `test_rlm_real_llm.py`, `test_planner.py`, `test_executor.py`, and the memory/retry e2e tests) are excluded from the deterministic run.

Run the deterministic suite:

```bash
python -m pytest tests/ -q \
  --ignore=tests/test_agent_memory_e2e.py \
  --ignore=tests/test_agent_retry_e2e.py \
  --ignore=tests/test_rlm_real_llm.py \
  --ignore=tests/test_langfuse.py \
  --ignore=tests/test_planner.py \
  --ignore=tests/test_agent_memory_integration.py \
  --ignore=tests/test_agent_evaluation.py \
  --ignore=tests/test_executor.py

python test_agent_contract.py     # set PYTHONIOENCODING=utf-8 on Windows
```

---

## Performance / Engineering Measurements

All figures below are **local deterministic measurements**, not benchmark scores and not live-model results.

**RLM parallel execution** — 3 simulated children, 0.30s of injected work each:

```
max_concurrency=1:  0.908s   peak concurrent = 1
max_concurrency=3:  0.305s   peak concurrent = 3
```

≈3× wall-clock reduction, with result ordering identical in both runs.

**Context compression** — synthetic 371k-shaped regression:

```
~400,483 estimated tokens -> ~30,921 estimated tokens   (1 pass, ~1.6 ms)
```

**Repository filtering:**

```
before:  6,652 files / 53.2 MB
after:     140 files /  0.5 MB
```

**Tool output limiting:**

```
bash `find . -type f`:  499,178 characters -> 20,000 characters
```

**Offline agent benchmark** (scripted model, 5 synthetic fixtures — measures execution behavior, not model quality): correctness was identical on both paths (5/5), while the RLM path used roughly 3× the model calls and tool calls of the normal path. On 2 of 5 tasks the RLM primary finding was incorrect even though the overall answer was correct. The benchmark reports these numbers as they are; it contains no heuristic favoring RLM.

---

## Project Structure

```
nanocode/
├── agent/                      # Single-agent pipeline
│   ├── agent.py                #   NanoCodeAgent: routing, retry/RSI loop, memory
│   ├── planner.py              #   Plan generation (LLM)
│   ├── executor.py             #   Tool-calling loop (LLM, streaming)
│   ├── evaluator.py            #   Deterministic outcome evaluation
│   ├── reflector.py            #   Deterministic diagnosis/improvement
│   ├── rsi.py                  #   RSIContext: what attempt N+1 learned from N
│   ├── context_budget.py       #   ContextBudgetManager, estimate_tokens, compression
│   ├── memory.py               #   Experience storage and keyword retrieval
│   ├── state.py                #   AgentState, identity/system-prompt boundary
│   └── tracer.py               #   Structured events (thread-safe)
│
├── rlm/                        # Recursive Language Model path
│   ├── router.py               #   normal vs rlm (deterministic)
│   ├── orchestrator.py         #   Decompose -> dispatch -> synthesize
│   ├── runtime.py              #   Budget reservation + bounded parallel execution
│   ├── decomposer.py           #   RLMDecomposer, DeterministicRLMDecomposer
│   ├── llm_decomposer.py       #   LLMRLMDecomposer (opt-in, validated, fallback)
│   ├── nanocode_handler.py     #   Child adapter, rate-limit retry, event forwarding
│   ├── synthesizer.py          #   Evidence-weighted synthesis
│   ├── evidence.py             #   Findings, grouping, conflicts, confidence
│   ├── budget.py               #   RLMBudget (depth / children / iterations)
│   ├── context.py              #   RLMContext
│   └── result.py, call.py, worker.py, controller.py, repl.py
│
├── tools/                      # Agent tools
│   ├── base.py                 #   Tool interface + OpenAI schema
│   ├── bash_tools.py           #   bash
│   ├── file_tools.py           #   read_file / write_file / edit_file
│   ├── grep_tool.py            #   grep (uses filtered traversal)
│   ├── ignore.py               #   Directory exclusion + iter_files()
│   ├── output_limit.py         #   20k-character output limiter
│   ├── todo_tool.py            #   todo_write
│   ├── spawn_agent_tool.py     #   task (sub-agent)
│   └── web_tools.py            #   web_fetch / web_search
│
├── cli/                        # Interactive terminal UI
│   ├── app.py, prompt.py, renderer.py, commands.py, banner.py,
│   └── session.py, workspace.py
│
├── config/settings.py          # Model, client, context and retry configuration
├── models/config.py            # AgentConfig, ToolCall
├── benchmark/                  # Offline benchmark (scripted model)
│   ├── runner.py, tasks.py, fixtures.py, fakes.py, scoring.py,
│   ├── metrics.py, report.py, context_scenarios.py, __main__.py
│
├── tests/                      # 34 test modules
├── DOCS/                       # Phase design documents
├── cli.py                      # One-shot entry point
├── run_nanocode.py             # Interactive entry point
├── test_agent_contract.py      # Standalone contract script
└── requirements.txt
```

- **`agent/`** — the single-agent pipeline and everything that bounds or improves one run.
- **`rlm/`** — the recursive path: routing, decomposition, bounded parallel execution, evidence synthesis.
- **`tools/`** — the agent's only means of touching the repository, plus the two layers that bound what it sees.
- **`cli/`** — the interactive terminal front end and the trace renderer.
- **`benchmark/`** — deterministic measurement of both paths using a scripted model; no network access.
- **`tests/`** — the deterministic suite; the project's primary correctness evidence.

---

## Installation

Developed and tested on **Python 3.12** (3.12.5 locally). No `pyproject.toml`/`setup.py` is present — the project runs from the repository directory.

```bash
git clone <repository-url>
cd nanocode

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# The interactive CLI additionally needs prompt_toolkit, which is not yet
# listed in requirements.txt:
pip install prompt_toolkit
```

Create a `.env` file in the repository root:

```bash
OPENROUTER_API_KEY=your-openrouter-key

# Optional — Langfuse tracing; spans are no-ops without these
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Optional — required only for the web_fetch / web_search tools
FIRECRAWL_API_KEY=your-firecrawl-key
```

Never commit `.env`. Modules **import** without credentials (a placeholder key keeps the client constructible so the offline test suite runs), but the CLI entry points call `require_api_key()` and fail immediately with a clear message if `OPENROUTER_API_KEY` is missing — no request is attempted.

---

## Configuration

`config/settings.py`:

| Name | Default | Purpose |
|---|---|---|
| `BASE_URL` | `https://openrouter.ai/api/v1` | Provider endpoint |
| `MODEL` | `poolside/laguna-s-2.1:free` | Model for planner, executor and LLM decomposition |
| `SDK_MAX_RETRIES` | `0` | SDK retries disabled — the application owns retry behavior |
| `MODEL_CONTEXT_LIMIT` | `262_144` | Documented provider window; the reason for the budget |
| `CONTEXT_SAFETY_MARGIN` | `0.35` | Headroom for the response, tool schemas and provider overhead |
| `CONTEXT_TOKEN_BUDGET` | `170_393` | What one request may actually spend (estimated tokens) |
| `CHARS_PER_TOKEN` | `4` | Estimation constant — an estimate, never an exact count |
| `MIN_RECENT_MESSAGES` | `6` | Recent messages never compressed |
| `MAX_COMPRESSED_ENTRY_CHARS` | `800` | Size a compressed entry is reduced to |
| `MAX_WEB_CONTENT_LENGTH` | `5000` | Cap on `web_fetch` content |

Other modules:

| Name | File | Default |
|---|---|---|
| `DEFAULT_MAX_OUTPUT_CHARS` | `tools/output_limit.py` | `20_000` |
| `DEFAULT_MAX_CONCURRENCY` | `rlm/runtime.py` | `3` |
| `RLMBudget.max_depth / max_children / max_iterations` | `rlm/budget.py` | `3 / 4 / 10` |
| `DEFAULT_DECOMPOSITION_STRATEGY` | `rlm/orchestrator.py` | `"deterministic"` |
| `DEFAULT_LLM_MAX_CHILDREN` | `rlm/llm_decomposer.py` | `3` |
| `MAX_TASK_CHARS` / `MAX_RESPONSE_CHARS` | `rlm/llm_decomposer.py` | `2_000` / `8_000` |
| `DEFAULT_THRESHOLD` | `rlm/router.py` | `0.5` (RLM routing score threshold) |
| `AgentConfig.max_iterations / max_retries` | `models/config.py` | `50 / 2` |

---

## Usage

### Interactive session

```bash
python run_nanocode.py
```

```
nanocode > Fix the authentication bug in auth.py
```

Project-wide phrasing routes to the RLM path:

```
nanocode > Find the authentication bug across the project
```

Built-in commands: `/help`, `/model`, `/mode`, `/status`, `/memory`, `/clear`, `/history`, `/exit`, `/quit`.

### One-shot task

```bash
python cli.py "Find the authentication bug across the project"
```

### Enabling LLM decomposition

There is no CLI flag for this; it is selected in code when constructing the orchestrator:

```python
from rlm.orchestrator import RLMOrchestrator

orchestrator = RLMOrchestrator(decomposition_strategy="llm")
```

### Development commands

```bash
# Offline agent benchmark (scripted model, no network)
python -m benchmark
python -m benchmark auth-expiry config-ttl

# Context-budget scenarios
python -m benchmark.context_scenarios

# Deterministic tests
python -m pytest tests/test_context_budget.py -v
```

---

## Example RLM Run

**User:** `Find the authentication bug across the project.`

```
Router          score >= 0.5  ->  strategy = rlm
   |
   v
Decomposer      3 child investigations
   |              1. core implementation and logic
   |              2. configuration, constants, environment
   |              3. call sites, tests, recent changes
   v
Runtime         budget reserved, up to 3 children run concurrently
   |
   v
Children        each: plan -> tools (grep, read_file) -> evaluate -> reflect
   |
   v
Synthesis       findings grouped, agreement counted, conflicts flagged
   |
   v
Final answer
```

The synthesized answer is structured rather than concatenated:

```
Primary finding:
auth.py validate_token() divides expires_at by 1000, so an expired token still passes.

Why: 2 independent child analyses (child 1, child 2) identified this issue.

Evidence:
- validate_token()
- auth.py

Other findings:
- [confirmed] SECRET_KEY is hardcoded in config.py. (child 3)
- [possible] The login flow probably needs a rate limiter. (child 3)

Confidence: high
```

The child count and the wording of each investigation depend on the decomposition strategy and the task: the deterministic decomposer produces 1–3 fixed perspectives, while the LLM decomposer produces task-specific investigations. If some children fail, the answer opens with an explicit partial-investigation notice.

---

## Known Limitations

- **Token estimation is approximate.** `estimate_tokens` divides characters by 4; it is not a tokenizer. Dense code or non-English text will estimate low. The 35% safety margin absorbs the error, but a real tokenizer would be strictly better.
- **Context compression is structural, not semantic.** Head/tail preservation can cut through the one line that mattered; nothing understands which output was important.
- **Dropped context cannot be recovered within a run.** The digest states how much was removed and suggests re-running a search, but the agent may repeat work.
- **Context budgeting is per request, not per run.** Fifty bounded requests still cost fifty requests' worth of tokens — this bounds context, not spend.
- **LLM decomposition quality is unbenchmarked.** Validation, fallback and safety are thoroughly tested against a fake client; whether dynamic children outperform the fixed perspectives is unmeasured.
- **Live LLM validation is pending.** Every result in this README is deterministic and offline.
- **Free model providers rate-limit.** The upstream shared-pool limit is provider-side; the code bounds and reports it rather than removing it.
- **RLM concurrency increases burst probability.** Three simultaneous requests raise 429 likelihood; `max_concurrency=1` restores sequential behavior.
- **Threads help I/O-bound model calls, not CPU-bound work.** The offline benchmark shows almost no gain precisely because the scripted model has no network latency.
- **No Agent Harness or Terminal-Bench result exists yet.**
- **The evidence synthesizer's primary-finding selection is heuristic** — it ranks a functional defect above a security or style observation by category, without knowing the user's question.
- **`prompt_toolkit` is missing from `requirements.txt`** despite being imported by the interactive CLI.

---

## Terminal-Bench / Evaluation

### Planned / Next — not yet run

**No Terminal-Bench evaluation has been performed, and no score is claimed.** Terminal-Bench 2.1 evaluation is planned but requires an agent harness that does not exist yet.

Intended evaluation flow:

```
NanoCode
   |
   v
Agent Harness            <- not yet implemented
   |
   v
Harbor / Terminal-Bench 2.1
   |
   v
Task execution
   |
   v
Task tests
   |
   v
Evaluation metrics
```

The existing `benchmark/` package measures *execution behavior* (model calls, tool calls, children, context sizes, correctness on synthetic fixtures) against a scripted model. It is deliberately not a standardized benchmark. Building the harness that lets NanoCode run real Terminal-Bench tasks against a real model is the next major engineering step.

---

## Roadmap

**Completed**

- ✓ Core coding agent (planner / executor / evaluator / reflector / memory)
- ✓ RLM routing
- ✓ RLM decomposition (deterministic default, LLM opt-in)
- ✓ Parallel RLM child execution
- ✓ RLM failure isolation
- ✓ Rate-limit handling
- ✓ RLM observability and CLI event forwarding
- ✓ Recursive self-improvement (RSI)
- ✓ Tool exploration filtering
- ✓ Tool output limiting
- ✓ Context budgeting and deterministic compression

**Next**

- → Agent Harness
- → Terminal-Bench 2.1 integration
- → Real-model validation
- → Benchmark-driven improvements
- → Production-quality evaluation and reporting

---

## Engineering Highlights

**1. 371k-token context explosion.** A real run assembled 371,405 input tokens against a 262,144-token limit; per-tool caps alone could not prevent it, because twenty bounded 20k results still sum past the window. A two-layer design — per-result truncation plus a request-level `ContextBudgetManager` doing role-preserving head/tail compression — reduced a synthetic reproduction from ~400,483 to ~30,921 estimated tokens in one pass and ~1.6 ms, with tool-call pairing intact.

**2. Recursive task decomposition.** A complex investigation exceeds what one linear agent run explores well. Decomposition was placed behind an `RLMDecomposer` interface with a deterministic strategy-based default, so an LLM-backed implementation could later be added as an opt-in strategy — with strict JSON validation, deduplication, budget capping and deterministic fallback — without changing the runtime, handler, synthesizer or router.

**3. Parallel child execution with deterministic ordering.** Sequential children made RLM roughly N× slower than a single run. Bounded `ThreadPoolExecutor` execution with budget reserved *before* dispatch, results written into pre-allocated slots by decomposition index, and identity assigned from decomposition order gives ≈3× wall-clock reduction on 3 children while keeping synthesis input byte-identical to the sequential case.

**4. RLM child failure isolation.** One child raising an exception aborted the entire investigation, discarding sibling work that had already succeeded. Converting child exceptions into failed `RLMResult` values in their own slots — and collecting every future rather than failing fast — means a success/failure/success run now completes all three children and synthesizes from the two that worked.

**5. Provider rate-limit amplification.** The OpenAI SDK silently retried 429s twice, turning one application call into three HTTP attempts into an already-saturated shared pool. Disabling SDK retries, classifying `RateLimitError` explicitly, and adding a bounded per-child retry inside the child's existing concurrency slot reduced a failing call from 3 HTTP attempts to 1 and made partial results visible instead of silently passing as complete.

**6. RSI-informed retries.** The original retry loop re-ran a failed attempt with the same planning context — repetition, not improvement. `RSIContext` threads the previous attempt's evaluation, reflection, attempt number and truncated response into the next attempt's prompt, with a deterministic guarantee that attempt N+1's input differs from attempt N's even when reflection produces nothing usable.

**7. Tool exploration of massive dependency directories.** Recursive search treated `.venv` and `node_modules` as user source, walking 6,652 files and 53.2 MB. A single centralized traversal helper that prunes generated and dependency directories — while leaving explicit path access untouched — reduced the same walk to 140 files and 0.5 MB.

**8. Context preservation under compression.** Naive truncation risks two failures: producing an invalid API request by orphaning a `tool_call`, and promoting tool text into a system instruction. Compression operates on assistant/tool *units* and never rewrites a role, verified by tests that inject `IGNORE ALL PREVIOUS INSTRUCTIONS…` into an old tool result and assert it never appears in any system message.

**9. End-to-end tracing of nested child agents.** Child agents recorded correct events into tracers nobody was subscribed to, so the CLI showed nothing during RLM runs while Langfuse showed everything. Forwarding real child events into the parent tracer — with child identity attached and lock-serialized delivery — made interleaved parallel child activity visible and correctly attributed without adding a single `print` to the RLM layer.

---

## Development Philosophy

- **Deterministic tests before live-model validation.** Every layer is provable offline; live validation is a separate, later step.
- **Bounded resource usage.** Output, context, concurrency, retries and recursion all have explicit configured limits.
- **Explicit failure handling.** Failures are classified, isolated and reported — never converted into silent success.
- **Observability.** Behavior is inspectable through structured events rather than print statements.
- **Separation of concerns.** The router decides *whether*, the decomposer decides *what*, the runtime decides *how much*, the child decides *how*.
- **Backward compatibility.** Existing contracts (`Tool.execute`, `RLMResult`, `RLMChildTask`, budget semantics) are preserved across phases.
- **No silent partial results.** A partial investigation says it is partial.
- **No unbounded context growth.** Every path to the model passes a budget check first.

---

## License

License: Not yet specified.
