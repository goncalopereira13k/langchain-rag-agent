from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
import bs4

load_dotenv()

# 1. Load
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 3. Store
embeddings = OllamaEmbeddings(model="llama3")

vector_store = PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)

print(f"Indexed {len(splits)} chunks into PGVector.")
