

import os

from dotenv import load_dotenv
load_dotenv()
from langfuse.openai import openai

BASE_URL = "https://openrouter.ai/api/v1"
# MODEL = "poolside/laguna-s-2.1:free"
MODEL="openrouter/free"
API_KEY = os.getenv("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MAX_WEB_CONTENT_LENGTH = 5000

# The OpenAI client refuses to construct without a key, which would make
# importing any agent module impossible offline; the deterministic test suites
# import agent modules without credentials. A placeholder keeps import working.
#
# It is deliberately not a usable key: it exists so that *importing* NanoCode
# is possible without credentials, not so that running it is. Entry points call
# require_api_key() so a real run still fails immediately, before any request
# is attempted.
MISSING_API_KEY = "missing-openrouter-api-key"

# The SDK retries 429s twice by default, so one application-level request
# became three HTTP attempts against an already rate-limited shared pool --
# retrying is exactly the wrong reflex there, and it was invisible to the
# tracer. Retries are owned by the application instead (see
# rlm/nanocode_handler.py), so there is one retry owner rather than two.
SDK_MAX_RETRIES = 0

client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY or MISSING_API_KEY,
    max_retries=SDK_MAX_RETRIES,
)


def require_api_key() -> None:
    """Fail fast when NanoCode is started without credentials."""

    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your environment or .env "
            "file before running NanoCode."
        )