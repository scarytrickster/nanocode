"""Tests for centralized tool-output size limiting.

Deterministic: real temp files and local commands, no LLM, no network.
"""

import sys

import pytest

from tools.bash_tools import BashTool
from tools.file_tools import ReadFileTool
from tools.grep_tool import GrepTool
from tools.ignore import iter_files
from tools.output_limit import (
    DEFAULT_MAX_OUTPUT_CHARS,
    limit_output,
    limit_tool_output,
)


TRUNCATION_MARKER = "[OUTPUT TRUNCATED]"


def payload(result: str) -> str:
    """The output text with any truncation notice removed."""

    return result.split(TRUNCATION_MARKER)[0].rstrip("\n")


# ---------------------------------------------------------------------------
# 1-3. Small, exact, and oversized output
# ---------------------------------------------------------------------------

def test_small_output_is_unchanged():

    text = "hello world"

    assert limit_tool_output(text, max_chars=100) == text

    result = limit_output(text, max_chars=100)

    assert result.truncated is False
    assert result.original_chars == 11
    assert result.returned_chars == 11


def test_output_exactly_at_the_limit_is_unchanged():

    text = "x" * 500

    assert limit_tool_output(text, max_chars=500) == text
    assert limit_output(text, max_chars=500).truncated is False


def test_output_above_the_limit_is_truncated():

    text = "x" * 501

    result = limit_output(text, max_chars=500)

    assert result.truncated is True
    assert payload(result.text) == "x" * 500


# ---------------------------------------------------------------------------
# 4-6. Truncation is explicit and bounded
# ---------------------------------------------------------------------------

def test_truncated_output_carries_an_explicit_indicator():

    result = limit_tool_output("y" * 5_000, max_chars=100)

    assert TRUNCATION_MARKER in result


def test_truncation_reports_original_and_returned_sizes():

    result = limit_output("z" * 184_532, max_chars=20_000)

    assert result.original_chars == 184_532
    assert result.returned_chars == 20_000

    # The counts are visible to the agent, not just in the metadata.
    assert "184,532" in result.text
    assert "20,000" in result.text


def test_returned_payload_is_bounded_by_the_limit():

    result = limit_output("q" * 1_000_000, max_chars=1_000)

    assert result.returned_chars == 1_000
    assert len(payload(result.text)) == 1_000

    # The whole result, notice included, stays a small multiple of the limit.
    assert len(result.text) < 1_000 + 500


def test_beginning_of_output_is_preserved():

    text = "FIRST LINE MATTERS\n" + "filler\n" * 10_000

    result = limit_tool_output(text, max_chars=1_000)

    assert result.startswith("FIRST LINE MATTERS\n")


# ---------------------------------------------------------------------------
# 7-10. Determinism, edges, unicode, memory
# ---------------------------------------------------------------------------

def test_repeated_calls_are_deterministic():

    text = "repeat me " * 10_000

    assert limit_tool_output(text, max_chars=777) == limit_tool_output(
        text, max_chars=777
    )


def test_empty_output_works():

    result = limit_output("", max_chars=100)

    assert result.text == ""
    assert result.truncated is False
    assert result.original_chars == 0
    assert result.returned_chars == 0


def test_none_is_treated_as_empty_output():

    assert limit_tool_output(None, max_chars=100) == ""


def test_unicode_text_is_handled_by_character_not_byte():

    text = "héllo wörld ünïcode ✅ 日本語テキスト"

    assert limit_tool_output(text, max_chars=1_000) == text

    result = limit_output(text * 1_000, max_chars=10)

    assert result.truncated is True
    assert payload(result.text) == text[:10]
    assert result.returned_chars == 10


def test_very_large_output_does_not_grow_the_returned_result():

    result = limit_output("a" * 5_000_000, max_chars=DEFAULT_MAX_OUTPUT_CHARS)

    assert result.original_chars == 5_000_000
    assert result.returned_chars == DEFAULT_MAX_OUTPUT_CHARS
    assert len(result.text) < DEFAULT_MAX_OUTPUT_CHARS * 2


def test_a_non_positive_limit_is_rejected():

    with pytest.raises(ValueError):
        limit_output("text", max_chars=0)


def test_default_limit_is_configurable_per_call():

    text = "b" * (DEFAULT_MAX_OUTPUT_CHARS + 1)

    assert limit_output(text).truncated is True
    assert limit_output(text, max_chars=DEFAULT_MAX_OUTPUT_CHARS * 2).truncated is False


# ---------------------------------------------------------------------------
# 11-12. BashTool
# ---------------------------------------------------------------------------

def big_stdout_command(chars: int, stream: str = "stdout") -> str:
    """A portable python one-liner that floods one stream."""

    target = "sys.stdout" if stream == "stdout" else "sys.stderr"

    return (
        f'"{sys.executable}" -c '
        f'"import sys; {target}.write(\'A\' * {chars})"'
    )


def test_bash_limits_large_stdout():

    result = BashTool().execute({"command": big_stdout_command(200_000)})

    assert TRUNCATION_MARKER in result
    assert len(payload(result)) == DEFAULT_MAX_OUTPUT_CHARS
    assert "200,000" in result


def test_bash_limits_large_stderr():

    result = BashTool().execute(
        {"command": big_stdout_command(200_000, stream="stderr")}
    )

    assert TRUNCATION_MARKER in result
    assert len(payload(result)) == DEFAULT_MAX_OUTPUT_CHARS


def test_bash_bounds_a_recursive_listing(tmp_path):

    for index in range(50):
        (tmp_path / f"file_{index}.txt").write_text("x" * 5_000, encoding="utf-8")

    result = BashTool().execute({"command": f'cat "{tmp_path}"/*.txt'})

    assert TRUNCATION_MARKER in result
    assert len(payload(result)) == DEFAULT_MAX_OUTPUT_CHARS


# ---------------------------------------------------------------------------
# 13-14. GrepTool and ReadFileTool
# ---------------------------------------------------------------------------

def test_grep_limits_a_large_number_of_matches(tmp_path):

    for index in range(40):
        (tmp_path / f"src_{index}.py").write_text(
            "NEEDLE = 1\n" * 500, encoding="utf-8"
        )

    result = GrepTool().execute({"pattern": "NEEDLE", "path": str(tmp_path)})

    assert TRUNCATION_MARKER in result
    assert len(payload(result)) == DEFAULT_MAX_OUTPUT_CHARS


def test_read_file_limits_a_large_file(tmp_path):

    target = tmp_path / "huge.log"
    target.write_text("L" * 200_000, encoding="utf-8")

    result = ReadFileTool().execute({"path": str(target)})

    assert TRUNCATION_MARKER in result
    assert len(payload(result)) == DEFAULT_MAX_OUTPUT_CHARS
    assert "200,000" in result


def test_reading_a_large_file_is_still_allowed(tmp_path):

    target = tmp_path / "huge.log"
    target.write_text("HEADER\n" + "L" * 200_000, encoding="utf-8")

    result = ReadFileTool().execute({"path": str(target)})

    # Explicit access is not blocked, only bounded.
    assert result.startswith("HEADER\n")
    assert not result.startswith("Error")


# ---------------------------------------------------------------------------
# 15-17. Normal-sized results are untouched
# ---------------------------------------------------------------------------

def test_small_bash_output_is_unchanged():

    result = BashTool().execute({"command": 'echo hello-from-bash'})

    assert result.strip() == "hello-from-bash"
    assert TRUNCATION_MARKER not in result


def test_small_grep_output_is_unchanged(tmp_path):

    (tmp_path / "auth.py").write_text("token = 'NEEDLE'\n", encoding="utf-8")

    result = GrepTool().execute({"pattern": "NEEDLE", "path": str(tmp_path)})

    assert result.endswith("token = 'NEEDLE'")
    assert TRUNCATION_MARKER not in result


def test_grep_with_no_matches_is_unchanged(tmp_path):

    (tmp_path / "auth.py").write_text("token = 1\n", encoding="utf-8")

    result = GrepTool().execute({"pattern": "NEEDLE", "path": str(tmp_path)})

    assert result == "No matches found."


def test_small_file_read_is_unchanged(tmp_path):

    target = tmp_path / "small.py"
    target.write_text("x = 1\n", encoding="utf-8")

    assert ReadFileTool().execute({"path": str(target)}) == "x = 1\n"


def test_read_error_messages_are_unchanged(tmp_path):

    result = ReadFileTool().execute({"path": str(tmp_path / "missing.py")})

    assert result.startswith("Error: File not found:")


# ---------------------------------------------------------------------------
# 18. Directory filtering from the previous phase is untouched
# ---------------------------------------------------------------------------

def test_directory_filtering_still_applies_with_output_limiting(tmp_path):

    (tmp_path / "auth.py").write_text("NEEDLE = 1\n", encoding="utf-8")

    vendored = tmp_path / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "api.py").write_text("NEEDLE = 1\n", encoding="utf-8")

    walked = list(iter_files(str(tmp_path)))

    assert not any(".venv" in path for path in walked)

    result = GrepTool().execute({"pattern": "NEEDLE", "path": str(tmp_path)})

    assert "auth.py" in result
    assert ".venv" not in result
