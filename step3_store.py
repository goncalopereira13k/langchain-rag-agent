import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Step 1 - Load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

# Step 2 - Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"Chunks to index: {len(splits)}")

# Step 3 - Store
# PGVector will: 1) embed each chunk with Ollama, 2) store the vectors in Postgres
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=os.getenv("DATABASE_URL"),
    pre_delete_collection=True,
)

document_ids = vector_store.add_documents(documents=splits)

print(f"Stored {len(document_ids)} documents.")
print(f"First 3 IDs: {document_ids[:3]}")
