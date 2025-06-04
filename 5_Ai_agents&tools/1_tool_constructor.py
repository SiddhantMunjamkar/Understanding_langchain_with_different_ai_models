# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace API token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file")
    hf_token = input("Please enter your HuggingFace API token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# Functions for the tools
def greet_user(name):
    """Greets the user by name."""
    return f"Hello, {name}!"


def reverse_string(text):
    """Reverses the given string."""
    return text[::-1]


def concatenate_strings(a_and_b):
    """Concatenates two strings. Format: 'string1,string2'"""
    try:
        a, b = a_and_b.split(',', 1)
        return a.strip() + b.strip()
    except ValueError:
        return "Error: Please provide two strings separated by a comma."


# Create tools using the Tool constructor approach
tools = [
    Tool(
        name="GreetUser",
        func=greet_user,
        description="Greets the user by name. Input should be a name.",
    ),
    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverses the given string. Input should be a string to reverse.",
    ),
    Tool(
        name="ConcatenateStrings",
        func=concatenate_strings,
        description="Concatenates two strings together. Input should be two strings separated by a comma, like 'hello,world'.",
    ),
]

# Initialize a HuggingFace model with more conservative settings
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=256,
    do_sample=False,
    huggingfacehub_api_token=hf_token,
    model_kwargs={
        "stop": ["Human:", "Assistant:"]
    },
    timeout=30
)

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create the agent using the create_structured_chat_agent function
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor with more conservative settings
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,
    return_intermediate_steps=False
)

# Test with a simple example first
try:
    print("Testing with a simple greeting...")
    response = agent_executor.invoke({"input": "Greet Alice"})
    print("Response:", response["output"])
    
    print("\nTesting with string reversal...")
    response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
    print("Response:", response["output"])
    
    print("\nTesting with string concatenation...")
    response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
    print("Response:", response["output"])
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
