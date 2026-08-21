"""A scripted stand-in for the model, so the benchmark runs offline.

Only the HTTP boundary is faked: `chat.completions.create` is replaced, and
everything above it -- Planner, Executor, the real tools, the router, the RLM
orchestrator, runtime, decomposer and evidence synthesis -- is the production
code path. That is the point: the benchmark measures the real architecture,
not a simulation of it.

The model does not know the answer. It greps the project, reads what the grep
pointed at, and reports lines that match a small suspicion heuristic. If the
fixture contains no defect it says so, which is how a false positive would
show up as an incorrect result.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field

# How the model reacts to a child task's wording. Different RLM children get
# different search terms, which is what makes them find different things.
FOCUS_PROBES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("configuration", "config", "constant", "environment", "dependencies"), "SECRET|TTL|DEBUG|CONFIG"),
    (("test", "tests", "recent changes", "call sites"), "def |import |assert"),
)

DEFAULT_PROBE = "token|session|time|expires"

# What the model treats as suspicious in source it has actually read. Each
# entry is (pattern, description). Nothing here names a fixture or a task.
SUSPICION_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"time\.time\(\)\s*\*\s*1000|\*\s*1000"),
        "multiplies a seconds value by 1000, mixing seconds and milliseconds",
    ),
    (
        re.compile(r"\b([A-Z][A-Z0-9_]*(?:TTL|TIMEOUT|EXPIR|LIFETIME)[A-Z0-9_]*)\s*=\s*0\b"),
        "is set to 0, so the value expires immediately",
    ),
    (
        re.compile(r"\b(SECRET_KEY|PASSWORD|API_KEY|TOKEN)\s*=\s*['\"]"),
        "is a hardcoded credential in source",
    ),
)

MAX_FILE_READS = 2

# A grep hit: "path:line: text". The path may carry a Windows drive letter,
# so the leading "C:" is matched explicitly rather than banned as a colon.
GREP_HIT_PATTERN = re.compile(
    r"^(?P<path>(?:[A-Za-z]:)?[^:]+):(?P<line>\d+):\s?(?P<text>.*)$"
)


@dataclass
class ModelCall:
    """One recorded call to the fake model."""

    stage: str
    model: str
    child_task: str = ""


@dataclass
class ScriptedModel:
    """Deterministic model behaviour over a real project directory."""

    project: str
    calls: list[ModelCall] = field(default_factory=list)

    # ------------------------------------------------------------------
    # OpenAI-compatible surface
    # ------------------------------------------------------------------

    def create(self, *, model, messages, stream=False, tools=None, **kwargs):
        task = self._task_of(messages)

        if not stream:
            self.calls.append(ModelCall(stage="planner", model=model, child_task=task))

            return self._plan_response()

        self.calls.append(ModelCall(stage="executor", model=model, child_task=task))

        return self._execute(messages, task)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_response(self):
        plan = (
            "1. Search the project for the relevant code.\n"
            "2. Read the files the search points at.\n"
            "3. Report what the source actually shows."
        )

        message = types.SimpleNamespace(content=plan)

        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, messages, task: str):
        observations = self._tool_outputs(messages)

        if not observations:
            return self._tool_call_stream(
                "grep",
                {"pattern": self._probe(task), "path": self.project},
            )

        candidates = self._candidate_files(observations[0])

        already_read = len(observations) - 1

        if already_read < min(MAX_FILE_READS, len(candidates)):
            return self._tool_call_stream(
                "read_file",
                {"path": candidates[already_read]},
            )

        return self._answer_stream(observations[1:], candidates)

    def _probe(self, task: str) -> str:
        lowered = task.lower()

        for markers, probe in FOCUS_PROBES:
            if any(marker in lowered for marker in markers):
                return probe

        return DEFAULT_PROBE

    def _candidate_files(self, grep_output: str) -> list[str]:
        """Files the grep actually matched, most matches first."""

        counts: dict[str, int] = {}

        for line in grep_output.splitlines():
            match = GREP_HIT_PATTERN.match(line.strip())

            if not match:
                continue

            path = match.group("path")

            if path.endswith(".py"):
                counts[path] = counts.get(path, 0) + 1

        return [
            path
            for path, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _answer_stream(self, file_contents: list[str], candidates: list[str]):
        findings: list[str] = []

        for path, content in zip(candidates, file_contents):
            findings.extend(self._inspect(path, content))

        if findings:
            answer = "\n".join(findings)
        elif candidates:
            answer = (
                "No defect was found in "
                + ", ".join(self._basename(path) for path in candidates[:MAX_FILE_READS])
                + "."
            )
        else:
            answer = "No relevant source was found for this task."

        return self._text_stream(answer)

    def _inspect(self, path: str, content: str) -> list[str]:
        """Report suspicious lines the file actually contains."""

        findings: list[str] = []

        name = self._basename(path)

        lines = content.splitlines()

        for index, line in enumerate(lines):
            for pattern, description in SUSPICION_PATTERNS:
                if not pattern.search(line):
                    continue

                symbol = self._enclosing_symbol(lines, index)

                location = f"{name} {symbol}" if symbol else name

                findings.append(
                    f"{location} line {index + 1}: `{line.strip()}` {description}."
                )

        return findings

    def _enclosing_symbol(self, lines: list[str], index: int) -> str:
        """The function containing a line, or the constant it assigns."""

        assignment = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", lines[index])

        for position in range(index, -1, -1):
            function = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[position])

            if function:
                return f"{function.group(1)}()"

        if assignment:
            return assignment.group(1)

        return ""

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rsplit("/", 1)[-1]

    # ------------------------------------------------------------------
    # Message helpers and stream construction
    # ------------------------------------------------------------------

    @staticmethod
    def _task_of(messages) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))

        return ""

    @staticmethod
    def _tool_outputs(messages) -> list[str]:
        return [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "tool"
        ]

    @staticmethod
    def _chunk(content=None, tool_calls=None, finish_reason=None):
        delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)

        choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)

        return types.SimpleNamespace(choices=[choice])

    def _text_stream(self, text: str):
        return iter([self._chunk(content=text), self._chunk(finish_reason="stop")])

    def _tool_call_stream(self, name: str, arguments: dict):
        import json

        call = types.SimpleNamespace(
            index=0,
            id=f"call_{name}",
            function=types.SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments),
            ),
        )

        return iter(
            [
                self._chunk(tool_calls=[call]),
                self._chunk(finish_reason="tool_calls"),
            ]
        )
