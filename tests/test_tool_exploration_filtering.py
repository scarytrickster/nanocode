"""Tests for exclusion of dependency/generated directories during exploration.

Deterministic: real temp directories, no LLM, no network.
"""

import os

import pytest

from tools.grep_tool import GrepTool
from tools.file_tools import ReadFileTool
from tools.ignore import (
    IGNORED_DIRECTORIES,
    iter_files,
    should_ignore_directory,
    should_ignore_path,
)


NEEDLE = "AUTH_TOKEN_MARKER"


@pytest.fixture
def project(tmp_path):
    """A project tree mixing real source with dependency/generated dirs."""

    def write(relative: str) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"value = '{NEEDLE}'\n", encoding="utf-8")

    # Real source, including a nested package.
    write("auth.py")
    write("app/main.py")
    write("app/services/login.py")

    # Dependency and generated directories, including nested ones.
    write(".venv/pyvenv.cfg")
    write(".venv/lib/site-packages/requests/api.py")
    write("node_modules/express/index.js")
    write("node_modules/nested/node_modules/deep/index.js")
    write(".git/config")
    write("__pycache__/auth.cpython-312.pyc")
    write("app/__pycache__/main.cpython-312.pyc")
    write("app/services/__pycache__/login.cpython-312.pyc")
    write("build/output.py")
    write("dist/bundle.js")
    write(".pytest_cache/CACHEDIR.TAG")
    write(".mypy_cache/data.json")
    write(".ruff_cache/content")
    write(".tox/py312/log.txt")
    write("venv/bin/activate")
    write("env/bin/activate")

    return tmp_path


def walked(root) -> list[str]:
    """Relative POSIX paths produced by the centralized walker."""

    return [
        os.path.relpath(path, root).replace("\\", "/")
        for path in iter_files(str(root))
    ]


def grep_hits(root) -> list[str]:
    """Relative POSIX paths reported by the grep tool."""

    output = GrepTool().execute({"pattern": NEEDLE, "path": str(root)})

    if output == "No matches found.":
        return []

    hits = []

    for line in output.splitlines():
        filepath = line.rsplit(":", 2)[0]
        hits.append(os.path.relpath(filepath, root).replace("\\", "/"))

    return hits


# ---------------------------------------------------------------------------
# 1-4. Excluded directories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "excluded",
    [".venv", "node_modules", ".git", "__pycache__"],
)
def test_recursive_exploration_excludes_directory(project, excluded):

    paths = walked(project)

    assert paths
    assert not any(path.startswith(f"{excluded}/") for path in paths)

    hits = grep_hits(project)

    assert not any(path.startswith(f"{excluded}/") for path in hits)


def test_every_ignored_directory_is_excluded(project):

    paths = walked(project)

    for ignored in IGNORED_DIRECTORIES:
        assert not any(
            part == ignored
            for path in paths
            for part in path.split("/")[:-1]
        ), ignored


# ---------------------------------------------------------------------------
# 5-6. Source stays reachable
# ---------------------------------------------------------------------------

def test_source_directories_remain_accessible(project):

    paths = walked(project)

    assert "app/main.py" in paths
    assert "app/services/login.py" in paths


def test_normal_file_is_still_returned(project):

    assert "auth.py" in walked(project)
    assert "auth.py" in grep_hits(project)


def test_only_source_files_are_explored(project):

    assert sorted(walked(project)) == [
        "app/main.py",
        "app/services/login.py",
        "auth.py",
    ]


# ---------------------------------------------------------------------------
# 7-8. Recursion and nesting
# ---------------------------------------------------------------------------

def test_filtering_works_recursively(project):

    paths = walked(project)

    # Generated dirs nested inside real source packages are pruned too.
    assert not any("__pycache__" in path for path in paths)

    # ...while their sibling source files survive.
    assert "app/services/login.py" in paths


def test_nested_excluded_directories_are_not_returned(project):

    paths = walked(project)

    assert not any("node_modules" in path for path in paths)
    assert not any("site-packages" in path for path in paths)


def test_deeply_nested_source_is_still_explored(tmp_path):

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "auth.py").write_text("x = 1\n", encoding="utf-8")

    (deep / "__pycache__").mkdir()
    (deep / "__pycache__" / "auth.pyc").write_text("junk\n", encoding="utf-8")

    assert walked(tmp_path) == ["a/b/c/auth.py"]


# ---------------------------------------------------------------------------
# 9. Explicit access is unchanged
# ---------------------------------------------------------------------------

def test_explicit_read_of_an_excluded_file_still_works(project):

    target = project / ".venv" / "pyvenv.cfg"

    content = ReadFileTool().execute({"path": str(target)})

    assert NEEDLE in content
    assert not content.startswith("Error")


def test_explicit_search_root_inside_an_excluded_directory_works(project):

    # The search root itself is never filtered: only traversal below it.
    paths = walked(project / ".venv")

    assert "pyvenv.cfg" in paths
    assert "lib/site-packages/requests/api.py" in paths

    assert grep_hits(project / ".venv")


def test_grep_of_an_excluded_root_reports_matches(project):

    output = GrepTool().execute(
        {"pattern": NEEDLE, "path": str(project / "node_modules")}
    )

    assert output != "No matches found."
    assert "index.js" in output


# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------

def test_ignored_directories_cover_the_required_set():

    required = {
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

    assert required <= IGNORED_DIRECTORIES


def test_should_ignore_directory():

    assert should_ignore_directory(".venv") is True
    assert should_ignore_directory("node_modules") is True

    assert should_ignore_directory("app") is False
    assert should_ignore_directory("auth.py") is False


def test_should_ignore_path_is_relative_to_the_root():

    assert should_ignore_path("./.venv/lib/api.py", root=".") is True
    assert should_ignore_path("./app/services/login.py", root=".") is False

    # Same file, but the search started inside the excluded directory.
    assert should_ignore_path(".venv/lib/api.py", root=".venv") is False


def test_grep_still_finds_project_source_in_this_repo():

    output = GrepTool().execute(
        {"pattern": "class GrepTool", "path": "tools"}
    )

    assert "grep_tool.py" in output
    assert ".venv" not in output
