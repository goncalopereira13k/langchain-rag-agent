from langchain.tools import tool

from src.rag.ingestion.store import get_vector_store

_vector_store = None


def _get_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    docs = _get_store().similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"<document>\n<source>{doc.metadata}</source>\n<content>{doc.page_content}</content>\n</document>"
        for doc in docs
    )
    return serialized, docs
