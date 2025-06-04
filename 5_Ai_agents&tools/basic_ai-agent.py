from dotenv import load_dotenv
import os
from langchain import hub
from langchain.agents import (AgentExecutor, create_react_agent)
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env
load_dotenv()

# Get HuggingFace API token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file")
    hf_token = input("Please enter your HuggingFace API token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


def get_current_time(*args, **kwargs):
    """Return the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time
    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format the time as desired


# list of tools
tools = [
    Tool(
        name="Get Current Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    )
]


prompt = hub.pull("hwchase17/react")


# Create a HuggingFace model for text generation
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    huggingfacehub_api_token=hf_token
)

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
   
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True)


# Run the agent with a test query
response = agent_executor.invoke({"input": "what time is it?"})
print(response)
