# RiveraGPT

RiveraGPT is an AI assistant platform and multi-model router built with FastAPI, LangChain, LangGraph, and Google Gemini, featuring active security guardrails, document RAG, persistent conversations, long-term memory, and tool integration.

![RiveraGPT UI](chat.png)

## Overview

RiveraGPT operates as an intelligent interface with capability-aware routing and system protection:
- **Capability-Aware Router**: Dynamically matches user intents (Speed, Deep Reasoning, Code Generation, Indic Languages) with fallback reliability across configured providers.
- **System-Level Security Rails**: Protects against direct and indirect prompt extraction, sanitizes output tokens to prevent credential leaks, and scrubs personally identifiable information (PII).
- **Document RAG (Retrieval-Augmented Generation)**: Upload and query documents (PDF, DOCX, TXT, MD, PY, CSV) embedded using Google Generative AI and ChromaDB vector store.
- **Conversational Memory**: SQLite-backed thread management supporting pin, rename, delete, and persistent long-term memory search.
- **Built-in Real-Time Tools**:
  - 🌦️ **Live Weather**: Dual-provider weather engine (WeatherAPI.com primary with automatic OpenWeatherMap fallback) providing real-time conditions, temperatures, humidity, wind, and forecasts.
  - 📰 **Breaking News**: Real-time news headlines and targeted topic queries powered by GNews.
  - 🗺️ **Directions & Navigation**: Turn-by-turn routing, distances, and duration calculations powered by HeiGIT OpenRouteService.
  - 💼 **Job Search**: Live job vacancy, company, and salary search powered by Adzuna.
  - 🧮 **Calculator**: Mathematical evaluation (arithmetic, trigonometry, logarithms, power, rounding).
  - 🌐 **Web Search**: Up-to-date web discovery via Tavily with GNews fallback.
  - 📄 **Document RAG**: In-memory and vector search for user-uploaded documents.
  - 🧠 **Memory Store**: Persistent recall and storage of user preferences and conversation facts.

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
├── model_router.py        # Intent detection, tool grounding & capability fallback
├── database.py            # SQLite database models, conversation CRUD & memory
├── rag.py                 # Document parsing, chunking, and ChromaDB retrieval
├── tools.py               # Live tools (weather, news, directions, jobs, math, RAG)
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
│   ├── test_router.py     # Intent classification and routing tests
│   └── test_tools.py      # Live tools and grounding tests
├── .env.example           # Documented configuration template
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Environment Configuration

Configure your API keys in `.env` (refer to `.env.example`):
- `GEMINI_API_KEY` / `GOOGLE_API_KEY`: Core LLM reasoning, code, and chat.
- `WEATHERAPI_KEY` / `OPENWEATHER_API_KEY`: Live weather conditions and forecasts.
- `GNEWS_API_KEY`: Real-time breaking news and headlines.
- `OPENROUTESERVICE_API_KEY`: Driving, cycling, and walking directions.
- `HUGGINGFACE_API_KEY` / `HF_TOKEN`: Hugging Face inference models.
- `ADZUNA_API_KEY` / `ADZUNA_APP_ID`: Job search and listings.
- `TAVILY_API_KEY`: General web search.
- `OPENAI_API_KEY` / `GROQ_API_KEY`: Alternative high-speed and flagship LLM providers.

## Deployment

### Vercel Deployment
This repository is configured for Vercel deployment with `vercel.json` and `api/index.py`:
1. Import your GitHub repository into [Vercel](https://vercel.com).
2. Under **Project Settings > Environment Variables**, add your API keys:
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Required for LLM and RAG embeddings)
   - `WEATHERAPI_KEY` / `OPENWEATHER_API_KEY` (For weather tools)
   - `GNEWS_API_KEY` (For live news)
   - `OPENROUTESERVICE_API_KEY` (For routing and directions)
   - `TAVILY_API_KEY` (For web search)
3. Deploy. All routes are proxied to `api/index.py` via FastAPI.

*Note: In serverless environments, local storage falls back to `/tmp`.*

### Container & Dedicated Server Deployment (Render / Railway / Docker)
For production deployments where persistent storage and long-lived connections are preferred:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

