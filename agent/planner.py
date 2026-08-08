"""
agent/planner.py

Creates an execution plan for a coding task.

The Planner:
- Reads the task from AgentState
- Asks the LLM to generate a step-by-step plan
- Stores the plan in AgentState

The Planner does NOT:
- Execute tools
- Modify files
- Run commands
- Execute the plan
"""

from __future__ import annotations
from pyexpat.errors import messages

from agent import tracer
from agent import state
from config.settings import MODEL, client
from agent.state import AgentState
from agent.tracer import Tracer


PLANNER_SYSTEM_PROMPT = """
You are the planning component of a terminal coding agent.

Your job is to create a concise, practical execution plan for the
user's coding task.

The plan should:

- contain concrete actions
- be ordered logically
- focus on solving the coding task
- be appropriate for a terminal environment
- avoid unnecessary explanation
- not execute any commands
- not modify any files

Return ONLY the plan.

Format the plan as numbered steps:

1. First step
2. Second step
3. Third step
"""


class Planner:
    """
    Creates an execution plan for an AgentState.
    """

    def __init__(self, tracer: Tracer | None = None):
        self.client = client
        self.tracer = Tracer()

    def run(self, state: AgentState) -> None:
        """Generate a plan and store it in state.plan."""

        messages = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": state.task,
            },
        ]

        with self.tracer.span(
            "planner",
            component="planner",
            task=state.task,
        ):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=False,
            )

            plan_text = response.choices[0].message.content or ""

            state.plan = self._parse_plan(plan_text)

    def _parse_plan(self, plan_text: str) -> list[str]:
        """
        Convert the LLM's numbered plan into a list of steps.
        """

        steps = []

        for line in plan_text.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove common numbered formats:
            #
            # 1. Run tests
            # 2) Inspect files
            #
            if line[0].isdigit():

                parts = line.split(".", 1)

                if len(parts) == 2:
                    step = parts[1].strip()

                    if step:
                        steps.append(step)
                    continue

                parts = line.split(")", 1)

                if len(parts) == 2:
                    step = parts[1].strip()

                    if step:
                        steps.append(step)
                    continue

            # Fallback: keep non-numbered lines
            steps.append(line)

        return steps