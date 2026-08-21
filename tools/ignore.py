"""Centralized exclusion rules for project exploration.

Recursive exploration (walking a project tree) must not descend into
dependency or generated directories: they are enormous, they are not the
user's source, and reading them explodes the context window.

Explicit access is a different thing and stays allowed: tools that receive a
concrete path act on that path. Only the *traversal* below a search root is
filtered, so `grep path=".venv"` still searches `.venv`, while `grep path="."`
skips it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator

# Directories excluded from recursive project exploration.
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".git",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
    }
)


def should_ignore_directory(
    name: str,
    ignored: Iterable[str] | None = None,
) -> bool:
    """True when a directory name is excluded from exploration."""

    ignored = IGNORED_DIRECTORIES if ignored is None else ignored

    return name in ignored


def should_ignore_path(
    path: str,
    root: str = ".",
    ignored: Iterable[str] | None = None,
) -> bool:
    """True when a path sits inside an excluded directory below `root`.

    The search root itself is never excluded: exploration is filtered relative
    to where it starts, so explicitly targeting an excluded directory works.
    """

    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        # Different drives on Windows: nothing to filter relative to root.
        return False

    parts = relative.replace("\\", "/").split("/")

    return any(
        should_ignore_directory(part, ignored)
        for part in parts
        if part not in ("", ".", "..")
    )


def iter_files(
    root: str,
    ignored: Iterable[str] | None = None,
) -> Iterator[str]:
    """Walk `root`, yielding file paths and pruning excluded directories.

    This is the single traversal helper for exploration tools so the filtering
    rules are not duplicated per tool.
    """

    for dirpath, dirnames, filenames in os.walk(root):

        # Pruning in place stops os.walk from descending, which is what keeps
        # nested excluded directories out of the results.
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not should_ignore_directory(name, ignored)
        )

        for filename in sorted(filenames):
            yield os.path.join(dirpath, filename)
