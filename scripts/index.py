import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.ingestion.loader import load_web_documents
from src.rag.ingestion.splitter import split_documents
from src.rag.ingestion.store import index_documents

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_web_documents(URLS)
    print(f"Loaded {len(docs)} document(s).")

    print("Splitting documents...")
    splits = split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Indexing into vector store...")
    ids = index_documents(splits, pre_delete_collection=True)
    print(f"Done. Stored {len(ids)} chunks.")
