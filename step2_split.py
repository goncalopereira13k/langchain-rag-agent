import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1 - Load (already know this)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

# Step 2 - Split
# chunk_size: max characters per chunk
# chunk_overlap: how many characters the next chunk shares with the previous one
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

print(f"Original documents: {len(docs)}")
print(f"Chunks after splitting: {len(splits)}")
print()
print(f"--- Chunk 1 ---")
print(splits[0].page_content)
print()
print(f"--- Chunk 2 ---")
print(splits[1].page_content)
print()
print(f"Last 200 chars of chunk 1 vs first 200 chars of chunk 2 (this is the overlap):")
print(f"\nEnd of chunk 1: ...{splits[0].page_content[-200:]}")
print(f"\nStart of chunk 2: {splits[1].page_content[:200]}...")
