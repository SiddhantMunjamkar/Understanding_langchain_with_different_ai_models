from dotenv import load_dotenv
import os
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceHub

# Load environment variables
load_dotenv()

# Get API token from environment
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not huggingface_api_token:
    huggingface_api_token = input("Please enter your HuggingFace API token: ")
    os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_api_token

# Create a Hugging Face chat model
model = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Using a more reliable model for text generation
    huggingfacehub_api_token=huggingface_api_token,
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

# Initialize chat history with system message
chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

def format_chat_history(messages):
    formatted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_messages.append(f"Assistant: {message.content}")
        elif isinstance(message, SystemMessage):
            formatted_messages.append(f"System: {message.content}")
    return "\n".join(formatted_messages) + "\nAssistant:"

# Chat loop
print("Chat with the AI (type 'exit' to end the conversation)")
print("-" * 50)

while True:
    try:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            break
        
        chat_history.append(HumanMessage(content=query))
        
        # Format the entire conversation history
        formatted_input = format_chat_history(chat_history)
        
        # Get the model's response
        result = model.invoke(formatted_input)
        
        # Clean and store the response
        response = result.strip()
        chat_history.append(AIMessage(content=response))
        
        print(f"\nAI: {response}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Let's continue with a new query.")

print("\n---- Chat History ----")
for message in chat_history:
    if isinstance(message, HumanMessage):
        print(f"\nYou: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")
    elif isinstance(message, SystemMessage):
        print(f"System: {message.content}")
