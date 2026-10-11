import os
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app import app
    from starlette.types import ASGIApp, Scope, Receive, Send

    class VercelPathRewriteMiddleware:
        def __init__(self, app: ASGIApp):
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope.get("type") == "http":
                raw_path = scope.get("path", "/")
                # Strip Vercel rewrite prefix so /api/index.py maps to /
                for prefix in ["/api/index.py", "/api"]:
                    if raw_path == prefix:
                        raw_path = "/"
                        break
                    elif raw_path.startswith(prefix + "/"):
                        raw_path = raw_path[len(prefix):]
                        break
                scope["path"] = raw_path or "/"
            await self.app(scope, receive, send)

    app.add_middleware(VercelPathRewriteMiddleware)

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
