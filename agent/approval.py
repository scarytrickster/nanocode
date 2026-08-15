from typing import Any


class ApprovalRejected(Exception):
    """Raised when a human rejects an agent action."""

    suppress_trace_failure = True



class ApprovalManager:
    """Handles human approval for agent actions."""

    VALID_MODES = {"auto", "confirm", "always"}

    def __init__(self, mode: str = "confirm") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid approval mode: {mode}. "
                f"Expected one of: {', '.join(sorted(self.VALID_MODES))}"
            )

        self.mode = mode

    def approve_plan(self, plan: Any) -> bool:
        """Request approval before executing an agent plan."""

        if self.mode == "auto":
            return True

        print("\nProposed Plan")
        print("-------------")
        print(plan)

        response = input(
            "\nApprove plan? [y/N/feedback]: "
        ).strip().lower()

        if response in {"y", "yes"}:
            return True

        if response in {"feedback", "f"}:
            feedback = input("Feedback: ").strip()

            # Feedback handling will be connected to the planner
            # in the next step.
            print(f"\nHuman feedback received: {feedback}")

        return False

    def approve_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Request approval before executing a tool."""

        if self.mode in {"auto", "confirm"}:
            return True

        print("\nTool Execution")
        print("--------------")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {arguments}")

        response = input(
            "\nExecute this tool? [y/N]: "
        ).strip().lower()

        return response in {"y", "yes"}

    def set_mode(self, mode: str) -> None:
        """Change the approval mode."""

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid approval mode: {mode}. "
                f"Expected one of: {', '.join(sorted(self.VALID_MODES))}"
            )

        self.mode = mode