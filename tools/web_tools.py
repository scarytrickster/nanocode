
import requests
from typing import Any

from tools.base import Tool
from config.settings import FIRECRAWL_API_KEY, MAX_WEB_CONTENT_LENGTH



class WebFetchTool(Tool):
    """Fetch a URL and return its content."""
    
    name = "web_fetch"
    description = "Fetch a URL and return its content as readable text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
        },
        "required": ["url"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"url": args["url"], "formats": ["markdown"]},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["data"]["markdown"][:MAX_WEB_CONTENT_LENGTH]
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {e}"


class WebSearchTool(Tool):
    """Search the web for information."""
    
    name = "web_search"
    description = "Search the web and return the top results (title, URL, description)."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/search",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"query": args["query"], "limit": 5, "sources": ["web"]},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()["data"]["web"]
            return "\n\n".join(
                f"{r['title']}\n{r['url']}\n{r.get('description', '')}" 
                for r in results
            )
        except requests.RequestException as e:
            return f"Error searching web: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing search results: {e}"