# Working of the Agent

This document explains exactly how the nanocode agent works when you run it, from startup to shutdown, including all internal processes and decision-making.

## Startup Sequence

### 1. Entry Point Execution
When you run `python agent.py`, the program begins executing:

1. **Banner Display** - Prints the startup banner showing version info and available commands
2. **Environment Loading** - Imports `load_dotenv()` to load environment variables from `.env` file
3. **Configuration Setup** - In `config/settings.py`:
   - Sets `BASE_URL` to OpenRouter API endpoint
   - Sets `MODEL` to the configured LLM model
   - Retrieves `API_KEY` and `FIRECRAWL_API_KEY` from environment
   - Creates OpenAI client instance
4. **Tools Initialization** - Calls `get_all_tools()` which:
   - Creates singleton instances of all 9 tool classes
   - Caches them in `_all_tools` global variable
   - Returns the complete tool list

### 2. Interactive Mode Setup
After initialization, the agent enters an infinite loop:

```
while True:
    prompt = "plan > " if plan_mode else "> "
    user_input = input(prompt)
    # Process input...
```

The agent presents either a normal prompt (`> `) or a plan-mode prompt (`plan > `) and waits for user input.

## Main Interaction Cycle

Each user interaction follows this cycle:

### Step 1: User Input Processing
1. User types a query and presses Enter
2. Special commands are checked first:
   - `/plan` toggles plan mode on/off
   - Empty input is ignored
3. Valid input is:
   - Added to the `messages` list as a `{"role": "user", "content": "..."}` entry
   - Passed to `run_agent()` with current tools and configuration

### Step 2: LLM Interaction
Inside `run_agent()`:

```python
stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tool_schemas,
    stream=True,
)
```

1. **Message Context** - The entire conversation history (system prompt + all previous exchanges) is sent
2. **Tool Schemas** - All tool function definitions are provided to the LLM
3. **Streaming Response** - Response comes back in chunks (streaming mode for real-time display)

### Step 3: Response Parsing
The `parse_tool_calls()` function processes the streaming response:

```python
def parse_tool_calls(stream):
    reply = ""
    tool_calls = []
    finish_reason = None
    
    for chunk in stream:
        # Handle content deltas
        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
            reply += choice.delta.content
        
        # Handle tool call deltas
        for tc in choice.delta.tool_calls or []:
            # Accumulate tool call info
            ...
    
    return reply, tool_calls, finish_reason
```

This function simultaneously:
- Prints response text to the terminal as it arrives
- Builds up the complete reply string
- Collects any tool call requests from the LLM
- Detects when the LLM wants to call tools (`finish_reason == "tool_calls"`)

### Step 4: Decision Making
Based on the LLM response, two paths are taken:

#### Path A: Direct Response
If the LLM provides text only (no tool calls):
- The reply is added to the conversation as an assistant message
- The function returns the reply to be displayed
- Control returns to the main loop

#### Path B: Tool Calls
If the LLM requests tool execution:
1. **Assistant Message Construction** - The tool calls are formatted as OpenAI API-compatible messages
2. **Tool Call Processing Loop** - Each requested tool is processed:

```python
for tc in tool_calls:
    # 1. Parse arguments from JSON
    args = json.loads(tc.arguments)
    
    # 2. Find the right tool
    tool = tools_by_name.get(tc.name)
    
    # 3. Check permissions
    # 4. Execute tool
    result = tool.execute(args)
    
    # 5. Add result back to conversation
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result,
    })
```

## Tool Execution Process

### Permission Checks
Before any tool executes, a series of checks occur:

1. **Plan Mode Check**
   ```python
   if config.plan_mode and not tool.is_read_only:
       # Block write operations in plan mode
   ```
   - In plan mode, ALL write operations are automatically blocked
   - Only read-only tools can execute

2. **Auto-Approval Check**
   ```python
   if not tool.is_read_only and not config.auto_approve:
       if not get_user_approval(tool, args):
           # User denied the operation
   ```
   - Write operations require explicit user approval unless auto-approved
   - Read-only tools skip this check

3. **User Approval Prompt**
   ```python
   answer = input(f"{tool.name}({json.dumps(args)}) [y/n] ")
   ```
   - Shows the tool name and arguments
   - Waits for user confirmation
   - Can be bypassed with Ctrl-C or EOF

### Tool Execution
After passing all checks:

1. **Call Tool Instance** - Uses the cached singleton tool instance
2. **Execute Method** - Calls `tool.execute(args)` with parsed arguments
3. **Result Capture** - Gets string result from tool execution
4. **Error Handling** - Wraps execution in try/except to handle any unexpected errors
5. **Message Formatting** - Converts any execution errors to readable messages

### Tool Categories
Different tool types have different behaviors:

| Tool Type | Examples | ReadOnly | Approval Required |
|-----------|----------|----------|-------------------|
| File Operations | read_file, write_file, edit_file | read_file: Yes<br>write_file/edit_file: No | Yes (for writes) |
| Search | grep | Yes | No |
| System | bash | No | Yes |
| Web | web_fetch, web_search | Yes | No |
| Task Management | todo_write | Yes | No |
| Agent | task | No | Yes |

## Multi-Turn Conversation Loop

After tool execution, the results are added back to the conversation:

```python
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,
    "content": result,
})
```

This allows the LLM to:
1. See the tool results
2. Continue reasoning based on the new information
3. Make additional tool calls if needed
4. Eventually provide a final answer

The loop continues up to `config.max_iterations` (default: 50) times, preventing infinite loops.

## Plan Mode Behavior

When plan mode is active (`/plan` toggle):

1. **Write Tool Blocking**
   ```python
   if config.plan_mode and not tool.is_read_only:
       messages.append({
           "tool_call_id": tc.id,
           "content": "Plan mode is on: write tools are disabled..."
       })
       continue  # Skip execution
   ```

2. **LLM Guidance** - The system prompt changes to encourage planning rather than direct action

3. **Safe Exploration** - Users can explore the codebase using read-only tools without risk of modifications

## Sub-Agent Spawning

The `task` tool creates isolated agent instances:

1. **Fresh Context** - Creates a completely new conversation history
2. **Filtered Tools** - Removes itself from the available tools to prevent recursion:
   ```python
   sub_tools = [t for t in get_all_tools() if t.name != self.name]
   ```
3. **Auto-Approval** - Runs with `auto_approve=True` for seamless execution
4. **Complete Execution** - Returns the final result after completion

This enables complex workflows where an agent can delegate tasks to specialized sub-agents.

## Error Handling

The agent implements comprehensive error handling:

1. **API Errors** - Caught during LLM call, logged, and returned as error messages
2. **Argument Parsing** - Invalid JSON arguments result in error messages
3. **Tool Resolution** - Unknown tools generate descriptive errors
4. **Tool Execution** - Exceptions during execution are caught and formatted
5. **Graceful Degradation** - Most errors are added back to conversation rather than crashing

## Session Management

### Conversation History
- All messages are stored in memory for the session duration
- No persistent session state between runs
- Each startup begins with a fresh conversation

### State Persistence
- Task lists persist within a session (via tool instance state)
- Files written to disk persist permanently
- No automatic session saving/resumption

## Shutdown

The agent exits cleanly on:
- `Ctrl-C` (KeyboardInterrupt)
- `Ctrl-D` (EOFError)
- Any fatal error

Displays a "Goodbye!" message before termination.

## Performance Characteristics

- **Streaming**: Responses appear in real-time as they're generated
- **Caching**: Tools and configuration are initialized once at startup
- **Memory**: Full conversation history retained (no rolling window)
- **Concurrency**: Single-threaded, sequential tool execution

## Monitoring and Debugging

Useful debugging observations:
- Tool calls show exact parameters being passed
- User approval prompts display tool names and arguments
- All tool results are visible in terminal output
- Error messages are descriptive and actionable
