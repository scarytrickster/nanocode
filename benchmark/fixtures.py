"""Tiny synthetic projects with known bugs.

Each fixture is a handful of small files written into a temporary directory.
The bug in each one is a real defect in the file content, so the expected
answer is derived from the fixture itself rather than from whatever the
implementation happens to produce.

Nothing is downloaded and nothing depends on buggy_auth_app existing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fixture:
    """A small project plus the defect it is known to contain."""

    name: str
    files: dict[str, str]

    def write(self, root: str) -> str:
        """Materialize the project under `root` and return the project path."""

        project = os.path.join(root, self.name)

        for relative, content in self.files.items():
            path = os.path.join(project, relative)

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)

        return project


# ---------------------------------------------------------------------------
# Fixture A -- single-file authentication expiry bug.
#
# validate_token compares a seconds-based timestamp against a milliseconds
# value, so every token looks expired.
# ---------------------------------------------------------------------------

AUTH_EXPIRY = Fixture(
    name="auth_expiry",
    files={
        "auth.py": (
            "import time\n"
            "\n"
            "\n"
            "def create_token(user):\n"
            "    expires_at = time.time() + 3600\n"
            "    return {'user': user, 'expires_at': expires_at}\n"
            "\n"
            "\n"
            "def validate_token(token):\n"
            "    now = time.time() * 1000\n"
            "    return token['expires_at'] > now\n"
        ),
        "README.md": "Demo authentication module.\n",
    },
)


# ---------------------------------------------------------------------------
# Fixture B -- configuration bug, with a security decoy.
#
# The functional defect is TOKEN_TTL_SECONDS = 0. The hardcoded secret is a
# real weakness but does not explain the reported failure.
# ---------------------------------------------------------------------------

CONFIG_TTL = Fixture(
    name="config_ttl",
    files={
        "config.py": (
            "SECRET_KEY = 'dev-secret-key'\n"
            "TOKEN_TTL_SECONDS = 0\n"
            "DEBUG = True\n"
        ),
        "app.py": (
            "from config import TOKEN_TTL_SECONDS\n"
            "\n"
            "\n"
            "def token_lifetime():\n"
            "    return TOKEN_TTL_SECONDS\n"
        ),
    },
)


# ---------------------------------------------------------------------------
# Fixture C -- multi-file data-flow bug.
#
# session.py hands milliseconds to an API that documents seconds.
# ---------------------------------------------------------------------------

SESSION_UNITS = Fixture(
    name="session_units",
    files={
        "clock.py": (
            "import time\n"
            "\n"
            "\n"
            "def seconds_now():\n"
            "    return time.time()\n"
        ),
        "session.py": (
            "from clock import seconds_now\n"
            "\n"
            "\n"
            "def start_session(store):\n"
            "    started_at = seconds_now() * 1000\n"
            "    store.save(started_at)\n"
        ),
        "store.py": (
            "class Store:\n"
            "    def save(self, started_at_seconds):\n"
            "        self.started_at = started_at_seconds\n"
        ),
    },
)


# ---------------------------------------------------------------------------
# Fixture D -- project-wide: the expiry bug plus unrelated observations.
# ---------------------------------------------------------------------------

PROJECT_WIDE = Fixture(
    name="project_wide",
    files={
        "auth.py": (
            "import time\n"
            "\n"
            "\n"
            "def validate_token(token):\n"
            "    now = time.time() * 1000\n"
            "    return token['expires_at'] > now\n"
        ),
        "config.py": (
            "SECRET_KEY = 'dev-secret-key'\n"
            "DEBUG = True\n"
        ),
        "handlers.py": (
            "from auth import validate_token\n"
            "\n"
            "\n"
            "def login(token):\n"
            "    if not validate_token(token):\n"
            "        return 'expired'\n"
            "    return 'ok'\n"
        ),
        "tests/test_login.py": (
            "from handlers import login\n"
            "\n"
            "\n"
            "def test_login_rejects_valid_token():\n"
            "    assert login({'expires_at': 1}) == 'expired'\n"
        ),
    },
)


# ---------------------------------------------------------------------------
# Fixture E -- no defect at all, used to check the benchmark does not reward
# a run that reports a bug where none exists.
# ---------------------------------------------------------------------------

CLEAN_MODULE = Fixture(
    name="clean_module",
    files={
        "calculator.py": (
            "def add(left, right):\n"
            "    return left + right\n"
            "\n"
            "\n"
            "def subtract(left, right):\n"
            "    return left - right\n"
        ),
    },
)


FIXTURES: dict[str, Fixture] = {
    fixture.name: fixture
    for fixture in (
        AUTH_EXPIRY,
        CONFIG_TTL,
        SESSION_UNITS,
        PROJECT_WIDE,
        CLEAN_MODULE,
    )
}
