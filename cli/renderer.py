from json import tool


class TerminalRenderer:
    def print_info(self, message: str) -> None:
        print(f"\n● {message}")

    def print_success(self, message: str) -> None:
        print(f"✓ {message}")

    def print_error(self, message: str) -> None:
        print(f"✗ {message}")

    def print_response(self, response: str) -> None:
        print(f"\n{response}")

    def print_command_result(self, result: str) -> None:
        print(f"\n{result}")

    def handle_trace(self, event) -> None:
        if event.name == "planner.started":
            self.print_info("Planning...")

        elif event.name == "planner.completed":
            self.print_success("Plan ready")

        elif event.name == "executor.started":
            self.print_info("Executing...")

        elif event.name == "executor.completed":
            self.print_success("Execution complete")

        elif event.name == "tool.started":
            tool = event.data.get("tool", "tool")
            self.print_tool_started(tool)

        elif event.name == "tool.completed":
            tool = event.data.get("tool", "tool")
            self.print_tool_completed(tool)

        elif event.name == "tool.failed":
            tool = event.data.get("tool", "tool")
            self.print_tool_failed(tool)

        elif event.name == "evaluator.completed":
            if event.data.get("success"):
                self.print_success("Task completed")
            else:
                self.print_error("Task evaluation failed")

        elif event.name == "reflector.completed":
            if event.data.get("should_improve"):
                self.print_info("Analyzing failure...")

        elif event.name == "retry.started":
            attempt = event.data.get("attempt", "?")
            print(f"↻ Retrying... attempt {attempt}")

        elif event.name == "retry.completed":
            self.print_success("Retry completed")

        elif event.name == "retry.exhausted":
            self.print_error("Retry limit reached")

        elif event.name.endswith(".failed"):
            self.print_error(
                f"{event.component} failed"
            )

    def print_tool_started(self, tool: str) -> None:
        print(f"  ↳ {tool}")


    def print_tool_completed(self, tool: str) -> None:
        print(f"  ✓ {tool}")


    def print_tool_failed(self, tool: str) -> None:
        print(f"  ✗ {tool}")