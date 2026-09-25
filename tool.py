from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
import re
from typing import Annotated
from dotenv import load_dotenv
load_dotenv()


tavily = TavilyClient(os.getenv("TAVILY_API_KEY"))

@tool

def web_search(query: str)-> str:
    """Search the web for information and return the first 5 results with snippets."""
    result = tavily.search(
        query=query,
        max_results=5
    )


    out = []

    for r in result['results']:
         out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    

    return "\n".join(out)


@tool
def scrape_url(
    url: Annotated[str, "The complete HTTP/HTTPS URL of the web page to scrape"]
) -> str:
    """Scrape and extract clean, readable text content from a public web page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. Remove non-content and noisy elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "svg",
        ]):
            element.decompose()

        # 2. Prefer semantic content containers if available
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(content|main|article)", re.I))
            or soup.find(class_=re.compile(r"(content|main|article)", re.I))
            or soup.body
            or soup
        )

        # 3. Extract text with separator to avoid merged words
        raw_text = main_content.get_text(separator="\n", strip=True)

        # 4. Collapse excessive whitespace and blank lines
        clean_text = re.sub(r"\n\s*\n+", "\n\n", raw_text)

        # 5. Cap output length to protect the LLM context window (e.g., ~12,000 chars)
        max_chars = 12000
        if len(clean_text) > max_chars:
            clean_text = (
                clean_text[:max_chars]
                + "\n\n[Content truncated due to context length limits...]"
            )

        return clean_text.strip() or "Error: No readable text found on page."

    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"
    except Exception as e:
        return f"Parsing Error: {e}"



# removed debug line