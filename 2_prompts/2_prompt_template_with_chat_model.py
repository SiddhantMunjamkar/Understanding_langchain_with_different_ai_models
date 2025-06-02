from dotenv import load_dotenv
import os
from langchain.prompts import ChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

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

print("---- Simple Joke Template ----")
# Part 1: Creating a prompt template
template = "Tell me a joke {topic}"
prompt_template = ChatMessagePromptTemplate.from_template(template, role="user")

prompt = prompt_template.format(topic="why was the cat sitting on the computer?")
response = model.invoke([prompt])
print("\nJoke:", response.content)

print("\n---- Prompt with Multiple Placeholders ----")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}
Assistant: """

prompt_multiple = ChatMessagePromptTemplate.from_template(template_multiple, role="user")
# Use format() instead of invoke()
prompt = prompt_multiple.format(
    adjective="funny",
    animal="cat"
)

result = model.invoke([prompt])
print("\nShort Story:", result.content)
