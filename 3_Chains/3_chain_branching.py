from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch

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

# define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",
     "Generate a thank you note for  this positive feedback :{feedback}"),
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",
     "Generate a response addressing this negative feedback :{feedback}"),
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",
     "Generate a request for more details for this neutral feedback :{feedback}"),
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",
     "Generate a message to escalate thsi feedback to a human agent :{feedback}"),
])

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human",
     "Classify the sentiment of this feedback as positive, negative  neutral ,or escalate :{feedback}"),
])
# Define the runnable branches for handling feedback

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain= classification_template | model | StrOutputParser()

#Combine classification and response generation into one chain
chain = classification_chain | branches

review= "The product is terrible . It broke after just one use and the quality is very poor."
result= chain.invoke({"feedback": review})
print(result)