import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Setup
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection=os.getenv("DATABASE_URL"),
)
model = init_chat_model("claude-sonnet-4-6")

# Retriever — interface para buscar documentos por similaridade
# k=2 significa que devolve os 2 chunks mais relevantes
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Prompt — instrui o modelo a responder apenas com base no contexto recuperado
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:

{context}

Question: {question}""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain com source documents
# RunnableParallel corre os dois ramos em paralelo:
#   - "context": busca os documentos relevantes
#   - "question": passa a pergunta diretamente
# .assign(answer=...) adiciona a resposta ao resultado final
rag_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(
    answer=(
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | model
        | StrOutputParser()
    )
)

# Query
result = rag_chain.invoke("What is Task Decomposition?")

print("=== ANSWER ===")
print(result["answer"])
print("\n=== SOURCE DOCUMENTS ===")
for doc in result["context"]:
    print(f"\nSource: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...")
