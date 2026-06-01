from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from src.rag.config import COLLECTION_NAME, DATABASE_URL, EMBEDDING_MODEL


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        pre_delete_collection=pre_delete_collection,
    )


def index_documents(splits: list[Document], pre_delete_collection: bool = True) -> list[str]:
    store = get_vector_store(pre_delete_collection=pre_delete_collection)
    return store.add_documents(documents=splits)
