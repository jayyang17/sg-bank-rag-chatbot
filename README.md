# sg-bank-rag-chatbot

**LangChain-powered RAG chatbot for querying DBS, OCBC, and UOB annual reports (2022–2023).**  
Modular, production-ready architecture with semantic search via ChromaDB and GPT-4 for contextual answers.

> Project is under active development — currently in **Phase 1: Data Ingestion**.

---

## Project Overview

This chatbot/API system enables users (investors, analysts, etc.) to ask natural language questions about Singapore banks' annual reports and receive concise, source-grounded responses.

Designed to:
- Showcase modern Retrieval-Augmented Generation (RAG) techniques
- Support local and cloud deployment
- Serve as a core portfolio project for LLM engineering

---

## Phase Progress

| Phase                         | Description |
|-------------------------------|-------------|
| Phase 1: Data Ingestion       | Extracted text, chunked, and tagged with metadata |
| Phase 2: Embedding & Indexing | Generate embeddings and store in Chroma vector DB |
| Phase 3: Retrieval & QA       | Build RetrievalQA chain using LangChain + GPT-4 |
| Phase 4: Frontend             | Flask interface or Telegram bot integration |
| Phase 5: Deployment           | Dockerized app with CI/CD to AWS EC2 |
| Phase 6: Evaluation           | Manual scoring, token tracking, latency benchmarks |

---

## 🧠 Planned Stack

- **LLM**: GPT-4 via OpenAI API
- **Embeddings**: `all-MiniLM-L6-v2` (`sentence-transformers`)
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **PDF Parsing**: pypdf, camelot
- **Frontend**: Flask, python-telegram-bot
- **Infra**: Docker, GitHub Actions, AWS EC2/S3

---

## 🛠️ Status

> ✅ Phase 1 complete — ingestion module working with metadata-tagged JSONL output.  
> ⏳ Moving next to embedding + Chroma vector store integration.

---

More commits dropping soon.
