# nanocode Architecture

## Overview

nanocode is a terminal-based AI coding agent that provides file operations, system interaction, web tools, and task management through an LLM-powered conversational interface.

## System Components

### Core Modules

```
nanocode
├── Configuration & Setup
├── Tool System
├── Agent Loop
└── User Interface
```

### 1. Configuration & Setup
- Loads environment variables from `.env` file
- Configures API endpoints (OpenRouter API)
- Sets up model parameters and limits
- Defines global tool instances (cached)

### 2. Tool System
A plugin-style architecture where each capability is implemented as a Tool class:

#### Tool Base Class
- Defines common interface for all tools
- Provides schema conversion for LLM function calling
- Tracks read-only vs. write operations

#### Available Tools
| Tool | Category | Description |
|------|----------|-------------|
| `read_file` | File I/O | Read files from disk |
| `write_file` | File I/O | Write content to files |
| `edit_file` | File I/O | Replace strings in files |
| `grep` | Search | Regex search across files |
| `bash` | System | Execute shell commands |
| `todo_write` | Task Mgmt | Manage task lists |
| `web_fetch` | Web | Fetch URLs via Firecrawl |
| `web_search` | Web | Search the web |
| `task` | Agent | Spawn sub-agents |

### 3. Agent Loop

#### Main Flow
1. User provides input via terminal
2. Input appended to conversation history
3. Stream response from LLM
4. Parse response for content and tool calls
5. Execute tool calls (with approval for write operations)
6. Append tool results to conversation
7. Repeat until completion

#### Special Modes
- **Plan Mode**: Disables write tools, encourages planning
- **Auto-Approve**: Bypasses user confirmation for write tools

### 4. User Interface
- Terminal-based interactive prompt
- Streaming response display
- Banner and status indicators
- Command shortcuts (`/plan`)

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            MAIN ENTRY POINT                            │
├─────────────────────────────────────────────────────────────────────────┤
│   Banner()     Init Config       Start Interactive Loop()             │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                           INTERACTIVE LOOP                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐       │
│  │ User Input  │───▶│ Message      │───▶│ Run Agent(messages) │       │
│  │ Processing  │    │ Management   │    │                     │       │
│  └─────────────┘    └──────────────┘    └─────────────────────┘       │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                           AGENT LOOP                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐        │
│  │ LLM Client   │◀──▶│ Conversation │◀──▶│ Tool Call Parser   │        │
│  │ (Streaming)  │    │ Manager      │    │                    │        │
│  └──────────────┘    └──────────────┘    └────────────────────┘        │
│        │                    │                  │                       │
│        ▼                    │                  ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐        │
│  │ Function     │    │ Tool Results │    │ Tool Execution     │        │
│  │ Schema       │    │ to Messages  │    │ Engine             │        │
│  └──────────────┘    └──────────────┘    └────────────────────┘        │
│        │                            │              │                  │
│        ▼                            │              ▼                  │
│  ┌──────────────┐                  │    ┌────────────────────┐        │
│  │ TOOL CLASSES │◀─────────────────┼────│ Approval Handler   │        │
│  │              │                  │    │ (Auto/Plan/User)   │        │
│  │ • read_file  │                  │    └────────────────────┘        │
│  │ • write_file │                  │              │                  │
│  │ • edit_file  │                  │              ▼                  │
│  │ • grep       │                  │    ┌────────────────────┐        │
│  │ • bash       │                  │    │ Read-Only Check    │        │
│  │ • todo_write │                  │    │ & Validation       │        │
│  │ • web_fetch  │                  │    └────────────────────┘        │
│  │ • web_search │                  │              │                  │
│  │ • task       │                  │              ▼                  │
│  └──────────────┘                  │    ┌────────────────────┐        │
│                                    │    │ External Services  │        │
│                                    │    │ • File System      │        │
│                                    │    │ • Shell Commands   │        │
│                                    │    │ • Web APIs         │        │
│                                    │    │ • Sub-Agents       │        │
│                                    │    └────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Error Handling Flow

```
┌─────────────┐
│ Tool Call   │
│ Attempt     │
└─────────────┘
       │
       ▼
┌─────────────┐  Yes ┌─────────────────┐
│ Error       │─────▶│ Catch Block     │
│ Occurred?   │      │ Execution       │
└─────────────┘      │                 │
       │ No          └─────────────────┘
       ▼                       │
┌─────────────┐                ▼
│ Success     │       ┌─────────────────┐
│ Result      │       │ Error Message   │
│ Returned    │       │ Formatted &     │
└─────────────┘       │ Added to        │
       │              │ Conversation    │
       ▼              └─────────────────┘
┌─────────────┐                │
│ Add Result  │◀──────────────┘
│ to Messages │
└─────────────┘
```

## Data Flow Diagram

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│   User      │    │ Conversation     │    │     LLM     │
│   Input     │───▶│ Context + Tools  │───▶│ API (Stream) │
└─────────────┘    └──────────────────┘    └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│Tool Executor│◀───│   Response       │◀───│ Tool Calls  │
│(with Auth)  │    │ Processing       │    │ Parsing     │
└─────────────┘    └──────────────────┘    └─────────────┘
     │                   │                        │
     ▼                   ▼                        ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│ Tool Result │───▶│Context Update│───▶│Assistant Response│──▶Terminal Output
└─────────────┘    └─────────────┘    └──────────────────┘
```

## Error Handling
- File operation errors (permissions, not found)
- API failures and timeouts
- Invalid tool parameters
- Tool execution exceptions

## Dependencies
- OpenAI SDK (LLM interactions)
- Requests (web operations)
- Firecrawl API (web scraping/search)
- Standard library modules

## Extensibility Points
- New Tool classes can be added to the tools registry
- Additional LLM providers can be integrated
- Custom system prompts per task type
- Middleware for conversation processing