from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = input("Please enter your OpenRouter API key: ")
    os.environ["OPENROUTER_API_KEY"] = api_key


# Define the directory containing the textfile and the presistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
presistent_directory = os.path.join(current_dir, "db", "chroma_db")


# check if the Chroma vector store aleady exists

if not os.path.exists(presistent_directory):
    print("Presistent directory does not exist. Initializing vector store...")

    # Ensure the text file exist
    if not os.path.exists(file_path):
        raise FileExistsError(
            f"The file {file_path} does not exist. Please check the path"
        )

    # Read the text content from the file with UTF-8 encoding
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n---Creating embeddings---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Using a model supported by OpenRouter
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LangChain Test"
        }
    )
    print("\n--Creating vector store---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=presistent_directory
    )

else:
    print("Vector store already exists. NO need to initialize")
