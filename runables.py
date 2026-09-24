import os
import sys
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Initialize the model
model = ChatMistralAI(
    model="codestral-2508", 
    api_key=os.getenv("MISTRALAI_API_KEY")
)

# 2. Create the prompt templates
shot_prompt = ChatPromptTemplate.from_template("explain {topic} in 1-2 sentences")
detail_prompt = ChatPromptTemplate.from_template("explain {topic} in detail")

# 3. Initialize the parser
parser = StrOutputParser()

# 4. Chain them together in parallel using RunnableParallel
chain = RunnableParallel({
    "shot": shot_prompt | model | parser,
    "detail": detail_prompt | model | parser
})

# 5. Invoke the chain
response = chain.invoke({
    "topic": "ml and dl v/s ai"
})

print("--- Short Explanation ---")
print(response["shot"])
print("\n--- Detailed Explanation ---")
print(response["detail"])