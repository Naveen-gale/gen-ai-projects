"""
agents.py  -  ResearchMind multi-agent definitions

Reads secrets from:
  Local:           .env file  (python-dotenv)
  Streamlit Cloud: App Settings > Secrets
"""
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tool import web_search, scrape_url


def _get_secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) or os.environ (local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, "")


def _build_llm():
    """Build LLM lazily so st.secrets is available at call time."""
    google_key  = _get_secret("GOOGLE_API_KEY")
    mistral_key = _get_secret("MISTRALAI_API_KEY")

    if google_key and google_key not in ("your_google_api_key_here", ""):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("LLM: Google Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_key,
            temperature=0.3,
        )

    if mistral_key and mistral_key not in ("your_mistral_api_key_here", ""):
        from langchain_mistralai import ChatMistralAI
        logger.info("LLM: Mistral open-mistral-nemo")
        return ChatMistralAI(
            model="open-mistral-nemo",
            api_key=mistral_key,
            temperature=0.3,
        )

    raise EnvironmentError(
        "No LLM API key found.\n"
        "Local: add GOOGLE_API_KEY or MISTRALAI_API_KEY to your .env file.\n"
        "Streamlit Cloud: add them in App Settings > Secrets."
    )


# Module-level cache (built once per process)
_llm = None
_writer_chain = None
_critic_chain  = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def _get_writer_chain():
    global _writer_chain
    if _writer_chain is None:
        _writer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research writer. Write clear, structured, and insightful reports."),
            ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual, and professional."""),
        ])
        _writer_chain = _writer_prompt | _get_llm() | StrOutputParser()
    return _writer_chain


def _get_critic_chain():
    global _critic_chain
    if _critic_chain is None:
        _critic_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sharp and constructive research critic. Be honest and specific."),
            ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
        ])
        _critic_chain = _critic_prompt | _get_llm() | StrOutputParser()
    return _critic_chain


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _invoke_with_backoff(fn, payload, max_retries: int = 4):
    delay = 8
    for attempt in range(1, max_retries + 1):
        try:
            return fn(payload)
        except Exception as exc:
            msg = str(exc).lower()
            is_transient = any(k in msg for k in ["429", "rate", "quota", "503", "502", "timeout"])
            if is_transient and attempt < max_retries:
                wait = delay * attempt
                logger.warning("Transient error (attempt %d/%d). Retrying in %ds...", attempt, max_retries, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_search_agent():
    logger.info("Building search agent...")
    return create_agent(model=_get_llm(), tools=[web_search])


def build_reader_agent():
    logger.info("Building reader agent...")
    return create_agent(model=_get_llm(), tools=[scrape_url])


def invoke_search_agent(agent, topic: str):
    return _invoke_with_backoff(agent.invoke, {
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })


def invoke_reader_agent(agent, topic: str, search_results: str):
    return _invoke_with_backoff(agent.invoke, {
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{search_results[:800]}"
        )]
    })


def invoke_writer_chain(topic: str, research: str):
    return _invoke_with_backoff(_get_writer_chain().invoke, {"topic": topic, "research": research})


def invoke_critic_chain(report: str):
    return _invoke_with_backoff(_get_critic_chain().invoke, {"report": report})