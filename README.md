# RiveraGPT

RiveraGPT is an AI assistant platform and multi-model router built with FastAPI, LangChain, LangGraph, and Google Gemini, featuring active security guardrails, document RAG, persistent conversations, long-term memory, and tool integration.

![RiveraGPT UI](chat.png)

## Overview

RiveraGPT operates as an intelligent interface with capability-aware routing and system protection:
- **Capability-Aware Router**: Dynamically matches user intents (Speed, Deep Reasoning, Code Generation, Indic Languages) with fallback reliability across configured providers.
- **System-Level Security Rails**: Protects against direct and indirect prompt extraction, sanitizes output tokens to prevent credential leaks, and scrubs personally identifiable information (PII).
- **Document RAG (Retrieval-Augmented Generation)**: Upload and query documents (PDF, DOCX, TXT, MD, PY, CSV) embedded using Google Generative AI and ChromaDB vector store.
- **Conversational Memory**: SQLite-backed thread management supporting pin, rename, delete, and persistent long-term memory search.
- **Built-in Assistant Tools**: Includes math evaluation, web search (Tavily), uploaded document retrieval, and memory recall.

## User-Facing Modes

RiveraGPT exposes unified abstract operating modes:
- **⚡ RiveraGPT Auto**: Automatic capability-aware intent routing.
- **⚡ RiveraGPT Fast**: Optimized for low latency and quick turnaround.
- **🧠 RiveraGPT Reasoner**: In-depth analytical reasoning and step-by-step logic.
- **💻 RiveraGPT Code**: Specialized code authoring, syntax checking, and debugging.
- **🇮🇳 RiveraGPT Indic**: Multilingual processing across Indian scripts and languages.

## Project Structure

```text
RiveraGPT/
├── app.py                 # FastAPI application routes and streaming SSE
├── agent.py               # LangGraph agent definitions and confidentiality prompt
├── model_registry.py      # Multi-model catalog and health tracker
├── model_router.py        # Intent detection and capability-aware fallback execution
├── database.py            # SQLite database models, conversation CRUD & memory
├── rag.py                 # Document parsing, chunking, and ChromaDB retrieval
├── tools.py               # Assistant tools (calculator, web search, memory, RAG)
├── security/              # Security and guardrails engine
│   ├── guardrails.py      # Input/output security pipeline
│   ├── injection.py       # Prompt extraction & injection defense
│   ├── llama_guard.py     # Content safety classification
│   ├── pii.py             # PII redaction (email, phone, SSN, card)
│   └── secrets.py         # Secret and credential masking
├── templates/             # Modern web interface
│   └── index.html
├── tests/                 # Automated test suite
│   ├── test_api.py        # Endpoint tests
│   ├── test_database.py   # Database lifecycle tests
│   ├── test_guardrails.py # Guardrails and scrubbing tests
│   └── test_router.py     # Intent classification and routing tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```



## Deployment

### Vercel Deployment
This repository is configured for Vercel deployment with `vercel.json` and `api/index.py`:
1. Import your GitHub repository into [Vercel](https://vercel.com).
2. Under **Project Settings > Environment Variables**, add your API keys:
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Required for LLM and RAG embeddings)
   - `TAVILY_API_KEY` (Optional, for web search tool)
3. Deploy. All routes are proxied to `api/index.py` via FastAPI.

*Note: In serverless environments, local storage falls back to `/tmp`.*

### Container & Dedicated Server Deployment (Render / Railway / Docker)
For production deployments where persistent storage and long-lived connections are preferred:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

