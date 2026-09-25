"""
tool.py  -  LangChain tools for ResearchMind
Exports: web_search, scrape_url
"""
import os
import re
import logging

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from typing import Annotated
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")


_tavily_key = _get_secret("TAVILY_API_KEY")
if not _tavily_key:
    raise EnvironmentError(
        "TAVILY_API_KEY is not set. Add it to .env or Streamlit secrets."
    )

_tavily = TavilyClient(_tavily_key)


@tool
def web_search(query: str) -> str:
    """Search the web for information and return the top 5 results with snippets."""
    try:
        result = _tavily.search(query=query, max_results=5)
        out = []
        for r in result.get("results", []):
            out.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['content'][:300]}\n"
            )
        return "\n".join(out) if out else "No results found."
    except Exception as e:
        logger.error("web_search error: %s", e)
        return f"Search Error: {e}"


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
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for el in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "svg"]):
            el.decompose()
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=re.compile(r"(content|main|article)", re.I))
            or soup.find(class_=re.compile(r"(content|main|article)", re.I))
            or soup.body
            or soup
        )
        raw = main.get_text(separator="\n", strip=True)
        clean = re.sub(r"\n\s*\n+", "\n\n", raw)
        if len(clean) > 12000:
            clean = clean[:12000] + "\n\n[Content truncated...]"
        return clean.strip() or "Error: No readable text found on page."
    except requests.exceptions.RequestException as e:
        logger.error("scrape_url request error %s: %s", url, e)
        return f"Request Error: {e}"
    except Exception as e:
        logger.error("scrape_url parse error %s: %s", url, e)
        return f"Parsing Error: {e}"