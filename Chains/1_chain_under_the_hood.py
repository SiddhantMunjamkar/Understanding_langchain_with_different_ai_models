from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = input("Please enter your OpenRouter API key: ")
    os.environ["OPENROUTER_API_KEY"] = api_key

# Setup OpenRouter + GPT-3.5-turbo
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=256,

)

# define prompt templates
prompt_tempelate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me a {joke_count} jokes.")
    ]
)


# Create the combined chain using Langchain Expression Language (LCEL)
chain = prompt_tempelate | model | StrOutputParser()

result= chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)
