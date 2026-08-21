from json import tool


class TerminalRenderer:
    def __init__(self) -> None:
        # Set for the duration of one event so RLM child activity is
        # attributable without every call site having to pass it along.
        self._prefix = ""

    def print_info(self, message: str) -> None:
        print(f"\n● {self._prefix}{message}")

    def print_success(self, message: str) -> None:
        print(f"✓ {self._prefix}{message}")

    def print_error(self, message: str) -> None:
        print(f"✗ {self._prefix}{message}")

    def print_response(self, response: str) -> None:
        print(f"\n{response}")

    def print_command_result(self, result: str) -> None:
        print(f"\n{result}")

    def handle_trace(self, event) -> None:
        self._prefix = self._child_prefix(event)

        try:
            self._render_trace(event)
        finally:
            self._prefix = ""

    def _child_prefix(self, event) -> str:
        """Identify which RLM child an event came from, when it came from one."""

        data = getattr(event, "data", {}) or {}

        child = data.get("rlm_child")

        if not child:
            return ""

        count = data.get("rlm_child_count") or 0

        return f"[RLM child {child}/{count}] " if count else f"[RLM child {child}] "

    def _render_trace(self, event) -> None:
        if event.name == "rlm.started":
            self.print_info("RLM: analyzing from multiple perspectives...")

        elif event.name == "rlm.decomposition.completed":
            children = event.data.get("children", 0)
            self.print_info(f"RLM: decomposed into {children} child tasks")

        elif event.name == "rlm.child.started":
            child = event.data.get("child", "?")
            of = event.data.get("of", "?")
            self.print_info(f"RLM child {child}/{of}")

        elif event.name == "rlm.child.retrying":
            child = event.data.get("child", "?")
            attempt = event.data.get("attempt", "?")
            print(f"↻ RLM child {child}: retrying (attempt {attempt})")

        elif event.name == "rlm.child.completed":
            child = event.data.get("child", "?")
            self.print_success(f"RLM child {child} complete")

        elif event.name == "rlm.child.failed":
            child = event.data.get("child", "?")
            error_type = event.data.get("error_type", "error")
            self.print_error(f"RLM child {child} failed ({error_type})")

        elif event.name == "rlm.synthesis.completed":
            successful = event.data.get("successful_children", 0)
            failed = event.data.get("failed_children", 0)

            if event.data.get("partial"):
                self.print_error(
                    f"RLM: partial result -- {successful} of "
                    f"{successful + failed} child analyses completed"
                )
            else:
                self.print_success("RLM: synthesis complete")

        elif event.name == "planner.started":
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
        print(f"  ↳ {self._prefix}{tool}")


    def print_tool_completed(self, tool: str) -> None:
        print(f"  ✓ {self._prefix}{tool}")


    def print_tool_failed(self, tool: str) -> None:
        print(f"  ✗ {self._prefix}{tool}")