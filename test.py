from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

print("OpenRouter Key Loaded:", API_KEY is not None)
print("Firecrawl Key Loaded:", FIRECRAWL_API_KEY is not None)