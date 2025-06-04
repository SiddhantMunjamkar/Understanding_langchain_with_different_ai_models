from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get HuggingFace API token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file")
    hf_token = input("Please enter your HuggingFace API token: ")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# Define Tools
def get_current_time():
    """Returns the current time in H:MM AM/PM format."""
    from datetime import datetime
    current_time = datetime.now().strftime("%I:%M %p")
    logger.info(f"Current time: {current_time}")
    return current_time


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        result = summary(query, sentences=2)
        logger.info(f"Wikipedia result for '{query}': {result}")
        return result
    except Exception as e:
        error_msg = f"I couldn't find any information on that. Error: {str(e)}"
        logger.error(f"Wikipedia search error for '{query}': {error_msg}")
        return error_msg


# Define the tools that the agent can use

# Define schema for Wikipedia tool
class WikipediaInput(BaseModel):
    query: str = Field(description="The search query to look up on Wikipedia")

tools = [
    StructuredTool.from_function(
        func=get_current_time,
        name="Time",
        description="Useful for when you need to know the current time.",
    ),
    StructuredTool.from_function(
        func=search_wikipedia,
        name="Wikipedia",
        description="Useful for when you need to know information about a topic.",
        args_schema=WikipediaInput,
    ),
]

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a HuggingFace model
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512,
    do_sample=True,
    top_k=1,
    top_p=0.9,
    repetition_penalty=1.2,
    huggingfacehub_api_token=hf_token,
    model_kwargs={
        "stop": ["Human:", "Assistant:", "User:"]
    }
)

# Create a structured Chat Agent with Conversation Buffer Memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# Create the agent with better error handling
agent = create_structured_chat_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt
)

# AgentExecutor with improved configuration
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=2,
    early_stopping_method="generate",
    return_intermediate_steps=False
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat Loop to interact with the user
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        # Add the user's message to the conversation memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Log the user input
        logger.info(f"User input: {user_input}")

        # Invoke the agent with the user input and the current chat history
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": memory.chat_memory.messages
            })
            bot_response = response["output"]
            
            # Log the bot response
            logger.info(f"Bot response: {bot_response}")
            
            print("Bot:", bot_response)

            # Add the agent's response to the conversation memory
            memory.chat_memory.add_message(AIMessage(content=bot_response))
        except Exception as e:
            error_message = f"I encountered an error while processing your request. Let me try to answer directly: {str(e)}"
            logger.error(error_message)
            
            # Try to use Wikipedia directly if the agent fails
            if "ambedkar" in user_input.lower():
                wiki_response = search_wikipedia("Dr. B.R. Ambedkar")
                print("Bot:", wiki_response)
                memory.chat_memory.add_message(AIMessage(content=wiki_response))
            else:
                print(f"Bot: {error_message}")
    except KeyboardInterrupt:
        print("\nBot: Session terminated by user.")
        break
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        print(f"Bot: {error_message}. Please try again.")
