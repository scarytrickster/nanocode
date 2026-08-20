import os

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client
from langfuse.openai import openai



BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "poolside/laguna-s-2.1:free"

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")


client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


print("Calling OpenRouter...")

response = client.chat.completions.create(
    name="langfuse-test",
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence.",
        }
    ],
    stream=False,
)

answer = response.choices[0].message.content

print("LLM response:")
print(answer)


print("Flushing Langfuse...")

langfuse = get_client()
langfuse.flush()

print("Done.")