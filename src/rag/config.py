import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "my_docs")
USER_AGENT: str = os.getenv("USER_AGENT", "rag-agent/1.0")

os.environ.setdefault("USER_AGENT", USER_AGENT)
