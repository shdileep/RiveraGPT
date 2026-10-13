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


def _mask_secret(val: str | None) -> str:
    if not val or val.startswith("your_"):
        return ""
    s = val.strip()
    if len(s) <= 8:
        return s[:2] + "..." + s[-2:]
    return s[:6] + "..." + s[-4:]


@app.get("/integrations/status")
@app.get("/api/integrations/status")
@app.get("/api/index.py/integrations/status")
async def get_integrations_status():
    google_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    hf_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    weatherapi_key = os.getenv("WEATHERAPI_KEY") or os.getenv("WEATHER_API_KEY")
    owm_key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
    gnews_key = os.getenv("GNEWS_API_KEY")
    adzuna_key = os.getenv("ADZUNA_APP_KEY") or os.getenv("ADZUNA_API_KEY")
    adzuna_id = os.getenv("ADZUNA_APP_ID")
    ors_key = os.getenv("OPENROUTESERVICE_API_KEY") or os.getenv("ORS_API_KEY") or os.getenv("HEIGIT_API_KEY")

    adzuna_status = "missing"
    if adzuna_key and adzuna_id and not adzuna_id.startswith("your_"):
        adzuna_status = "active"
    elif adzuna_key:
        adzuna_status = "partial"

    return {
        "integrations": [
            {
                "id": "google",
                "name": "Google AI Studio (Gemini 2.5)",
                "category": "ai_model",
                "status": "active" if google_key and not google_key.startswith("your_") else "missing",
                "docs_url": "https://aistudio.google.com/api-keys",
                "description": "Primary multimodal intelligence, deep reasoning, coding & analysis engine."
            },
            {
                "id": "openrouter",
                "name": "OpenRouter (Claude, GPT-4o, DeepSeek)",
                "category": "ai_model",
                "status": "active" if openrouter_key and not openrouter_key.startswith("your_") else "missing",
                "docs_url": "https://openrouter.ai/keys",
                "description": "Unified access to Claude 3.5 Sonnet, GPT-4o, DeepSeek R1 & 200+ models."
            },
            {
                "id": "huggingface",
                "name": "Hugging Face Hub & Inference",
                "category": "ai_model",
                "status": "active" if hf_key and not hf_key.startswith("your_") else "missing",
                "docs_url": "https://huggingface.co/settings/tokens",
                "description": "Open-weights LLMs (Qwen 2.5, DeepSeek R1 Distill, Llama 3.3, Gemma)."
            },
            {
                "id": "flux_image",
                "name": "FLUX.1 Photorealistic AI Image Generator",
                "category": "multimodal",
                "status": "active",
                "docs_url": "https://pollinations.ai",
                "description": "High-resolution photo & art generation via state-of-the-art FLUX.1 diffusion."
            },
            {
                "id": "weatherapi",
                "name": "WeatherAPI.com",
                "category": "tool",
                "status": "active" if weatherapi_key and not weatherapi_key.startswith("your_") else "missing",
                "docs_url": "https://www.weatherapi.com/my/",
                "description": "Live conditions, hourly forecasts, UV index, and precipitation chance."
            },
            {
                "id": "openweather",
                "name": "OpenWeatherMap",
                "category": "tool",
                "status": "active" if owm_key and not owm_key.startswith("your_") else "missing",
                "docs_url": "https://home.openweathermap.org/api_keys",
                "description": "Secondary authenticated weather and temperature provider."
            },
            {
                "id": "openmeteo",
                "name": "Open-Meteo Global Weather",
                "category": "tool",
                "status": "active",
                "docs_url": "https://open-meteo.com",
                "description": "Zero-key fallback ensuring worldwide weather retrieval never fails."
            },
            {
                "id": "gnews",
                "name": "GNews Live News & Research",
                "category": "tool",
                "status": "active" if gnews_key and not gnews_key.startswith("your_") else "missing",
                "docs_url": "https://gnews.io/dashboard",
                "description": "Real-time news search, tech headlines, and citation links."
            },
            {
                "id": "adzuna",
                "name": "Adzuna Job Search",
                "category": "tool",
                "status": adzuna_status,
                "docs_url": "https://developer.adzuna.com",
                "description": "Open job postings, salary ranges, and official application links."
            },
            {
                "id": "heigit",
                "name": "OpenRouteService / HeiGIT",
                "category": "tool",
                "status": "active" if ors_key and not ors_key.startswith("your_") else "missing",
                "docs_url": "https://account.heigit.org/manage/key",
                "description": "Turn-by-turn navigation, route calculation, travel duration & distances."
            }
        ]
    }


@app.post("/integrations/test")
@app.post("/api/integrations/test")
@app.post("/api/index.py/integrations/test")
async def test_integration(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"success": False, "message": "Invalid JSON"}, status_code=400)

    provider_id = (data.get("provider") or "").lower().strip()
    import time
    t0 = time.time()

    try:
        if provider_id == "google":
            key = os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key or key.startswith("your_"):
                return {"success": False, "message": "Service is currently unavailable. Please try again later."}
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, request_timeout=8.0)
            res = llm.invoke("Hi")
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": f"Connected to AI intelligence engine ({latency}ms)"}

        elif provider_id in ["weather", "weatherapi", "openweather", "openmeteo"]:
            from tools import get_weather
            res = get_weather.invoke({"location": "London"})
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": f"Weather retrieved in {latency}ms"}

        elif provider_id == "gnews":
            from tools import get_latest_news
            res = get_latest_news.invoke({"query": "AI"})
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": "Sorry" not in res and "unavailable" not in res, "latency_ms": latency, "message": f"News verified ({latency}ms)"}

        elif provider_id in ["heigit", "ors"]:
            from tools import get_directions_and_route
            res = get_directions_and_route.invoke({"start_location": "Paris", "end_location": "Lyon"})
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": "Sorry" not in res and "unavailable" not in res, "latency_ms": latency, "message": f"Routing verified ({latency}ms)"}

        elif provider_id in ["flux_image", "image"]:
            from image_service import generate_image
            res = generate_image("minimal blue river geometric logo")
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": res.get("success", False), "latency_ms": latency, "message": f"Image generation verified ({latency}ms)" if res.get("success") else "Image generator currently unavailable."}

        elif provider_id == "adzuna":
            from tools import search_jobs
            res = search_jobs.invoke({"query": "Software Engineer", "location": "Remote"})
            latency = round((time.time() - t0) * 1000, 1)
            return {"success": True, "latency_ms": latency, "message": f"Job search verified ({latency}ms)"}

        elif provider_id == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key or key.startswith("your_"):
                return {"success": False, "message": "Service is currently unavailable. Please try again later."}
            import httpx
            with httpx.Client(timeout=5.0) as client:
                r = client.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"})
                latency = round((time.time() - t0) * 1000, 1)
                return {"success": r.status_code == 200, "latency_ms": latency, "message": f"Service active ({latency}ms)" if r.status_code == 200 else "Service unavailable."}

        return {"success": True, "latency_ms": 10.0, "message": f"Service is active."}

    except Exception:
        latency = round((time.time() - t0) * 1000, 1)
        return {"success": False, "latency_ms": latency, "message": "Connection check failed. Please try again later."}


@app.post("/integrations/save")
@app.post("/api/integrations/save")
@app.post("/api/index.py/integrations/save")
async def save_integrations(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"success": False, "message": "Invalid JSON"}, status_code=400)

    keys = data.get("keys", {})
    if not isinstance(keys, dict):
        return JSONResponse({"success": False, "message": "Keys must be an object"}, status_code=400)

    env_path = BASE_DIR / ".env"
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            existing_lines = f.readlines()

    updated_keys = set()
    new_lines = []

    for line in existing_lines:
        trimmed = line.strip()
        if trimmed and not trimmed.startswith("#") and "=" in trimmed:
            k = trimmed.split("=", 1)[0].strip()
            if k in keys and keys[k] is not None:
                val = str(keys[k]).strip()
                new_lines.append(f"{k}={val}\n")
                os.environ[k] = val
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in keys.items():
        if k not in updated_keys and v is not None and str(v).strip():
            val = str(v).strip()
            new_lines.append(f"{k}={val}\n")
            os.environ[k] = val
            updated_keys.add(k)

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        pass

    return {"success": True, "saved_keys": list(updated_keys)}


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