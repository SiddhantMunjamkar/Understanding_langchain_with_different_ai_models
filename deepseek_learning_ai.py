from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load .env variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = input("Please enter your OpenRouter API key: ")
    os.environ["OPENROUTER_API_KEY"] = api_key

# Setup OpenRouter + DeepSeek
model = ChatOpenAI(
    model="openai/gpt-3.5-turbo",  # Using GPT-3.5-turbo through OpenRouter
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=256
)

# Initial system message
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

# Chat loop
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        chat_history.append(HumanMessage(content=user_input))
        response = model.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content))
        print("AI:", response.content)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure your API key is correct and try again.")
        break
