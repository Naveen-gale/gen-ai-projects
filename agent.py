from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv



load_dotenv()
llm = ChatMistralAI(model="ministral-3b-2512", api_key=os.getenv("MISTRALAI_API_KEY"))

# 1st agent 
def web_serch_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# 2nd agent
def web_scrape_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# promt 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])


w_chin = writer_prompt | llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
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


c_chin = critic_prompt | llm | StrOutputParser()




print(w_chin.invoke("The impact of AI on the world"))
print(c_chin.invoke("The impact of AI on the world"))