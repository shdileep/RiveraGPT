import os
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app import app as fastapi_app

    async def app(scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            # Vercel provides the original client path in x-matched-path or x-forwarded-uri
            matched_header = headers.get(b"x-matched-path") or headers.get(b"x-forwarded-uri")
            if matched_header:
                try:
                    real_path = matched_header.decode("utf-8")
                    if "?" in real_path:
                        real_path = real_path.split("?", 1)[0]
                    if real_path and real_path != "/api/index.py":
                        scope["path"] = real_path
                        scope["raw_path"] = real_path.encode("ascii")
                except Exception:
                    pass
            else:
                raw_path = scope.get("path", "/")
                for prefix in ["/api/index.py", "/api"]:
                    if raw_path == prefix:
                        raw_path = "/"
                        break
                    elif raw_path.startswith(prefix + "/"):
                        raw_path = raw_path[len(prefix):]
                        break
                scope["path"] = raw_path or "/"
                scope["raw_path"] = (raw_path or "/").encode("ascii")

        await fastapi_app(scope, receive, send)

except Exception as e:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    err_msg = traceback.format_exc()
    print(f"CRITICAL BOOT ERROR: {err_msg}")
    app = FastAPI(title="RiveraGPT Initialization Fallback")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def fallback_error(path: str):
        return JSONResponse(
            {
                "status": "error",
                "message": "RiveraGPT failed to initialize during serverless function boot.",
                "error": str(e),
                "hint": "Ensure required environment variables (e.g., GEMINI_API_KEY) are set in the Vercel Dashboard under Project Settings > Environment Variables.",
                "traceback": err_msg
            },
            status_code=500
        )
