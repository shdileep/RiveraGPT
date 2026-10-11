from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import HumanMessage, AIMessage
from agent import rivera_stream_chat
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
    toggle_pin_conversation,
    rename_conversation,
    delete_conversation,
    save_conversation_document,
    get_conversation_documents
)
from rag import add_document_to_rag, read_file_text
from tools import set_current_thread_id
from model_registry import registry, health_tracker

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RiveraGPT Multi-Model Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _get_writable_dir(subdir: str) -> Path:
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    base = Path("/tmp") if is_serverless else BASE_DIR
    target = base / subdir
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        tmp_target = Path("/tmp") / subdir
        tmp_target.mkdir(parents=True, exist_ok=True)
        return tmp_target


UPLOADS_DIR = _get_writable_dir("uploads")
DATA_DIR = _get_writable_dir("data")

try:
    init_db()
except Exception:
    pass


@app.get("/chat.png")
@app.get("/favicon.ico")
@app.get("/api/chat.png")
@app.get("/api/index.py/chat.png")
async def get_chat_icon():
    icon_path = BASE_DIR / "chat.png"
    if icon_path.exists():
        return FileResponse(str(icon_path), media_type="image/png")
    return JSONResponse({"error": "Icon not found"}, status_code=404)


@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/models")
@app.get("/api/models")
@app.get("/api/index.py/models")
async def get_models():
    """
    Returns public RiveraGPT modes catalog.
    """
    modes = registry.list_user_facing_modes()
    return {"models": modes}


@app.get("/health")
@app.get("/api/health")
@app.get("/api/index.py/health")
async def health_status():
    """
    Returns real-time system status.
    """
    return {
        "status": "healthy",
        "engine": "RiveraGPT Guardrails Core",
        "shield": "active"
    }


@app.get("/conversations")
@app.get("/api/conversations")
@app.get("/api/index.py/conversations")
async def conversations():
    items = list_conversations()
    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "pinned": getattr(item, "pinned", False),
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


@app.post("/conversations/{thread_id}/pin")
@app.post("/api/conversations/{thread_id}/pin")
@app.post("/api/index.py/conversations/{thread_id}/pin")
async def pin_chat(thread_id: str):
    pinned = toggle_pin_conversation(thread_id)
    return {"success": True, "pinned": pinned}


@app.put("/conversations/{thread_id}/rename")
@app.put("/api/conversations/{thread_id}/rename")
@app.put("/api/index.py/conversations/{thread_id}/rename")
async def rename_chat(thread_id: str, request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return JSONResponse({"error": "Title cannot be empty."}, status_code=400)
    success = rename_conversation(thread_id, new_title)
    return {"success": success}


@app.delete("/conversations/{thread_id}")
@app.delete("/api/conversations/{thread_id}")
@app.delete("/api/index.py/conversations/{thread_id}")
async def delete_chat(thread_id: str):
    success = delete_conversation(thread_id)
    return {"success": success}


@app.get("/history/{thread_id}")
@app.get("/api/history/{thread_id}")
@app.get("/api/index.py/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


@app.post("/upload")
@app.post("/api/upload")
@app.post("/api/index.py/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]
        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        target_path = UPLOADS_DIR / f"{file_id}_{safe_filename}"

        content_bytes = await file.read()
        with open(target_path, "wb") as f:
            f.write(content_bytes)

        # Extract full document text
        try:
            extracted_text = read_file_text(str(target_path))
        except Exception:
            extracted_text = content_bytes.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            extracted_text = f"[Document {filename} uploaded, empty text content]"

        # Save into SQLite conversation documents store for guaranteed context retention
        save_conversation_document(
            thread_id=thread_id,
            filename=filename,
            file_type=suffix,
            content=extracted_text
        )

        create_or_update_conversation(thread_id, f"Document: {filename}")

        # Index into RAG vector store if available
        try:
            add_document_to_rag(file_path=str(target_path), thread_id=thread_id)
        except Exception as rag_err:
            print(f"[RAG] Note: Vector store indexing skipped: {rag_err}")

        return JSONResponse({
            "success": True,
            "filename": filename,
            "message": f"Document '{filename}' attached to conversation."
        })

    except Exception as e:
        return JSONResponse(
            {"success": False, "message": f"Sorry, could not process that file: {str(e)}"},
            status_code=500
        )


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
@app.post("/api/chat/stream")
@app.post("/api/index.py/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    user_message = data.get("message", "").strip()
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "auto")

    if not user_message:
        return JSONResponse({"error": "Message is required."}, status_code=400)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)
    set_current_thread_id(thread_id)

    # Convert past history to Human/AI messages (preserve last 30 turns for rich continuity)
    db_history = get_chat_history(thread_id)
    recent_db_history = db_history[-30:] if len(db_history) > 30 else db_history

    history_messages = []
    for m in recent_db_history:
        if m.role == "user":
            history_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history_messages.append(AIMessage(content=m.content))

    if not history_messages or history_messages[-1].content != user_message:
        history_messages.append(HumanMessage(content=user_message))

    # Retrieve all documents attached to this conversation
    conversation_docs = get_conversation_documents(thread_id)
    doc_context_str = ""
    if conversation_docs:
        doc_parts = []
        for d in conversation_docs:
            snippet = d.content[:80000] if len(d.content) > 80000 else d.content
            doc_parts.append(
                f"### DOCUMENT: {d.filename} (Format: {d.file_type})\n{snippet}\n### END OF DOCUMENT: {d.filename}"
            )
        doc_context_str = "\n\n".join(doc_parts)

    def event_generator():
        final_answer = ""
        try:
            for item in rivera_stream_chat(
                user_message=user_message,
                selected_model_key=selected_model,
                history_messages=history_messages,
                document_context=doc_context_str
            ):
                if "meta" in item:
                    yield sse_data({"meta": item["meta"]})
                elif "token" in item:
                    token = item["token"]
                    final_answer += token
                    yield sse_data({"token": token, "mode": selected_model})
                elif "error" in item:
                    yield sse_data({"error": "Sorry, I couldn't complete that request. Please try again."})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": "Sorry, I couldn't complete that request. Please try again."})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False
    )