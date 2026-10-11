# MehadiAI

MehadiAI is an intelligent AI assistant application built with FastAPI, LangChain, LangGraph, and Google Gemini. It supports conversational chat, document upload, retrieval-augmented generation (RAG), memory, web search, and calculator tools. The project is designed to be easy to run locally and simple to deploy on Render.

## Features

- Chat with an AI assistant powered by Gemini models
- Upload documents such as PDF, DOCX, TXT, MD, PY, and CSV files
- Ask questions about uploaded files using RAG-based document search
- Maintain conversation history and thread-based chat sessions
- Use built-in tools for:
  - calculator
  - web search
  - memory saving and recall
  - uploaded document search
- FastAPI backend with streaming chat responses

## Live Demo

- Render Deployment: https://mehadiai.onrender.com

## Tech Stack

- Backend: FastAPI, Uvicorn
- AI Frameworks: LangChain, LangGraph
- LLM: Google Gemini
- Vector Database: Chroma
- Database: SQLite with SQLAlchemy
- Document Processing: PyPDF, docx2txt
- Templating: Jinja2

## Project Structure

```text
MehadiAI/
├── app.py                 # FastAPI application and routes
├── agent.py               # LangGraph agent setup and Gemini model integration
├── database.py            # SQLite database models and helpers
├── rag.py                 # Document loading and RAG logic
├── tools.py               # AI tools such as calculator, memory, and search
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── templates/             # HTML templates for the frontend
│   └── index.html
├── uploads/               # Uploaded documents are stored here
├── data/                  # SQLite databases and checkpoints
├── chroma_db/             # Chroma vector database storage
└── test.py                # Basic test file
```



## License

This project is available for personal and educational use. Please check the repository license before commercial use.
