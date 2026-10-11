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

## Prerequisites

Before running the project, make sure you have:

- Python 3.11 installed
- Conda or virtualenv available
- A Google Gemini API key
- A Tavily API key if you want web search functionality

## Environment Variables

Create a file named `.env` in the project root and add your configuration:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
GEMINI_MODEL=gemini-2.5-flash
```

> If your setup uses a different variable name for your Gemini API key, adjust it accordingly.

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Mehadii-Hassan/MehadiAI.git
cd MehadiAI
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n MehadiAI python=3.11 -y
conda activate MehadiAI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the environment file

Create a `.env` file in the project root and add the required environment variables.

### 5. Run the application

```bash
python app.py
```

### 6. Open the app in your browser

Visit:

```text
http://127.0.0.1:8080
```

You should now be able to chat with the assistant, upload documents, and ask questions about them.

## How to Use the Application

1. Open the app in your browser.
2. Start a conversation in the chat interface.
3. Upload a document from the upload area.
4. Ask questions about the uploaded file.
5. Use the AI tools for web search, memory, or calculations.

## Deploy on Render

To deploy this project on Render, follow these steps:

### 1. Push the project to GitHub

Make sure your repository is connected to GitHub.

### 2. Create a new Web Service on Render

- Go to Render Dashboard
- Click New > Web Service
- Connect your GitHub repository

### 3. Configure the build and start commands

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 4. Add environment variables in Render

Add the same environment variables from your `.env` file in the Render dashboard:

- GOOGLE_API_KEY
- TAVILY_API_KEY
- GEMINI_MODEL

### 5. Deploy

Click Deploy and wait for Render to build and start the service.

### 6. Open your live app

After deployment finishes, Render will provide a public URL. Replace the placeholder in the Live Demo section with your deployed link.

## Notes

- The app stores uploaded files in the `uploads/` folder.
- Chroma vector data is stored in the `chroma_db/` folder.
- Conversation history and memory are stored locally in the `data/` folder.
- For production deployment, ensure your API keys are configured securely in the hosting platform.

## License

This project is available for personal and educational use. Please check the repository license before commercial use.
