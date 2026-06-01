import bs4
from langchain_community.document_loaders import WebBaseLoader

# Fetch the blog post HTML and parse it with BeautifulSoup.
# bs_kwargs filters to only the CSS classes that contain the actual post —
# this strips out nav, sidebar, footer, etc. so we get clean text.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    },
)

docs = loader.load()

# loader.load() always returns a list — one Document per URL
print(f"Number of documents loaded: {len(docs)}")
print(f"Type of each item: {type(docs[0])}")
print()
print(f"Metadata: {docs[0].metadata}")
print()
print(f"First 500 characters of content:\n{docs[0].page_content[:500]}")
