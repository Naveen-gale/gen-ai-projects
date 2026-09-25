import os
import time
import logging
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tool import web_search, scrape_url
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

_api_key = os.getenv("MISTRALAI_API_KEY")
if not _api_key:
    raise EnvironmentError("MISTRALAI_API_KEY is not set in .env")

# mistral-small-latest - available on free/developer tier
# max_retries handled by tenacity wrapper below
llm = ChatMistralAI(model="mistral-small-latest", api_key=_api_key, temperature=0.3)


def _is_rate_limit(exc):
    """Return True if the exception is a 429 rate-limit error."""
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _invoke_with_retry(agent_or_chain, payload):
    """Invoke an agent or chain with automatic exponential-backoff retry on 429."""
    return agent_or_chain.invoke(payload)


def build_search_agent():
    logger.info("Building search agent...")
    return create_agent(model=llm, tools=[web_search])


def build_reader_agent():
    logger.info("Building reader agent...")
    return create_agent(model=llm, tools=[scrape_url])


def invoke_search_agent(agent, topic):
    return _invoke_with_retry(agent, {
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })


def invoke_reader_agent(agent, topic, search_results):
    return _invoke_with_retry(agent, {
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{search_results[:800]}"
        )]
    })


def invoke_writer_chain(topic, research):
    return _invoke_with_retry(writer_chain, {"topic": topic, "research": research})


def invoke_critic_chain(report):
    return _invoke_with_retry(critic_chain, {"report": report})


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

writer_chain = _writer_prompt | llm | StrOutputParser()

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

critic_chain = _critic_prompt | llm | StrOutputParser()