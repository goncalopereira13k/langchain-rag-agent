from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")

model = init_chat_model("claude-sonnet-4-6")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)