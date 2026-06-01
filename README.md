# LangChain RAG Agent

A question-answering application built with LangChain that uses Retrieval Augmented Generation (RAG) to answer questions about unstructured text sources.

## What it does

This project demonstrates two approaches to RAG:

- **RAG Agent** — orchestrates retrieval using a tool-calling agent; good general-purpose implementation
- **RAG Chain** — two-step pipeline with a single LLM call per query; fast and effective for simple queries

The demo app answers questions about [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.

## How it works

1. **Indexing** — ingests content from the source, splits it into chunks, embeds them, and stores them in a vector store
2. **Retrieval & Generation** — at query time, retrieves relevant chunks and passes them to the model to generate an answer

## Stack

- [LangChain](https://docs.langchain.com) — orchestration
- [Claude](https://anthropic.com) (`claude-sonnet-4-6`) — language model
- [Ollama](https://ollama.com) (`llama3`) — embeddings
- [LangSmith](https://smith.langchain.com) — tracing

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env` and fill in your keys:
   ```bash
   cp .env .env.local
   ```

   Required variables:
   ```
   ANTHROPIC_API_KEY=...
   LANGSMITH_API_KEY=...
   LANGSMITH_TRACING=true
   ```

3. Run:
   ```bash
   python init_chat_model.py
   ```
