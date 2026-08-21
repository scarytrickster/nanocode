"""Benchmark tasks and the criteria a correct answer has to satisfy.

Expected criteria describe the defect that is actually present in the fixture
(the file it lives in, the symbol that contains it, and the behaviour that
gives it away). They are written from the fixture source, never from what an
implementation produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Task categories.
SIMPLE_QUESTION = "simple-question"
SINGLE_FILE_BUG = "single-file-bug"
MULTI_FILE_BUG = "multi-file-bug"
PROJECT_INVESTIGATION = "project-investigation"
COMPARISON = "comparison"

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"


@dataclass(frozen=True)
class ExpectedFinding:
    """What an answer must identify to count as correct.

    `files` and `symbols` locate the defect; `behaviors` are the observable
    traces of it in the source. An empty ExpectedFinding means the fixture has
    no defect, and reporting one is wrong.
    """

    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    behaviors: tuple[str, ...] = ()
    should_find_defect: bool = True

    @property
    def criteria_count(self) -> int:
        return sum(
            1
            for group in (self.files, self.symbols, self.behaviors)
            if group
        )


@dataclass(frozen=True)
class BenchmarkTask:
    """One task, run identically down the normal and the RLM path."""

    id: str
    name: str
    prompt: str
    category: str
    fixture: str
    expected: ExpectedFinding = field(default_factory=ExpectedFinding)
    difficulty: str = MEDIUM


TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        id="auth-expiry",
        name="Authentication expiry bug",
        prompt="Find the authentication bug in this project.",
        category=SINGLE_FILE_BUG,
        fixture="auth_expiry",
        expected=ExpectedFinding(
            files=("auth.py",),
            symbols=("validate_token",),
            behaviors=("1000", "time.time"),
        ),
        difficulty=EASY,
    ),
    BenchmarkTask(
        id="config-ttl",
        name="Configuration token lifetime bug",
        prompt="Find the configuration bug that makes tokens expire immediately.",
        category=SINGLE_FILE_BUG,
        fixture="config_ttl",
        expected=ExpectedFinding(
            files=("config.py",),
            symbols=("TOKEN_TTL_SECONDS",),
            behaviors=("0",),
        ),
        difficulty=EASY,
    ),
    BenchmarkTask(
        id="session-units",
        name="Cross-file unit mismatch",
        prompt="Find the unit mismatch bug across the project.",
        category=MULTI_FILE_BUG,
        fixture="session_units",
        expected=ExpectedFinding(
            files=("session.py",),
            symbols=("start_session",),
            behaviors=("1000",),
        ),
        difficulty=MEDIUM,
    ),
    BenchmarkTask(
        id="project-wide",
        name="Project-wide authentication investigation",
        prompt="Find the authentication bug across the entire codebase.",
        category=PROJECT_INVESTIGATION,
        fixture="project_wide",
        expected=ExpectedFinding(
            files=("auth.py",),
            symbols=("validate_token",),
            behaviors=("1000",),
        ),
        difficulty=HARD,
    ),
    BenchmarkTask(
        id="clean-module",
        name="Module with no defect",
        prompt="Investigate whether the calculator module has a bug.",
        category=SIMPLE_QUESTION,
        fixture="clean_module",
        expected=ExpectedFinding(should_find_defect=False),
        difficulty=EASY,
    ),
)


TASKS_BY_ID: dict[str, BenchmarkTask] = {task.id: task for task in TASKS}
