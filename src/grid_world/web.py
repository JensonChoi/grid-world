from __future__ import annotations

from pathlib import Path
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from grid_world.config import RUNS_DIR


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Imagination Gridworld")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/rollouts")
    def rollouts() -> JSONResponse:
        path = RUNS_DIR / "latest" / "rollouts.json"
        if not path.exists():
            return JSONResponse(
                {
                    "error": "No rollout artifact found. Run collect-data, train-world-model, train-controller, and evaluate first."
                },
                status_code=404,
            )
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    return app


def serve_main() -> None:
    uvicorn.run("grid_world.web:create_app", factory=True, host="127.0.0.1", port=8000, reload=False)
