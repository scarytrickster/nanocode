Phase 2 flow

We're going to insert a Planner:

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
Planner.run(state)
│
▼
state.plan
│
▼
Executor.run(state)
│
├── LLM
├── Tool calls
├── Tool execution
└── Update messages
│
▼
Final response

The important part is:

Planner
│
▼
AgentState.plan
│
▼
Executor

The planner doesn't execute anything.

FINAL PHASE 2 :-

                         USER
                           │
                           ▼
                  NanoCodeAgent.run()
                           │
                           ▼
                    create_state()
                           │
                           ▼
                      AgentState
                           │
                           ▼
                       Planner
                           │
                           ▼
                      state.plan
                           │
                           ▼
                       Executor
                           │
                           ▼
                  _build_messages()
                           │
                    ┌──────┴──────┐
                    │             │
               state.messages  state.plan
                    │             │
                    └──────┬──────┘
                           ▼
                          LLM
                           │
                           ▼
                     Tool Calls?
                       /       \
                     Yes        No
                      │          │
                      ▼          ▼
                 Execute Tool   Final Answer
                      │
                      ▼
                Update Messages
                      │
                      └──────► LLM

FINAL V2

User
│
▼
NanoCodeAgent
│
├── creates shared Tracer
│
▼
Planner
│
├── planner.started
└── planner.completed
│
▼
AgentState.plan
│
▼
Executor
│
├── executor.started
├── llm.request
├── llm.response
├── tool.started
├── tool.completed
└── executor.completed
│
▼
Final Response
