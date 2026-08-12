

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
# MODEL = "poolside/laguna-s-2.1:free"
MODEL="openrouter/free"
API_KEY = os.getenv("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MAX_WEB_CONTENT_LENGTH = 5000

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)