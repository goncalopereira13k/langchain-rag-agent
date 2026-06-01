import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_web_documents(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(
        web_paths=tuple(urls),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    return loader.load()
