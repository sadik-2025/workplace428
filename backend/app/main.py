import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .database import Base, engine
from .routers import auth_router, formula_router, models_router, predict_router, stats_router

# Creates users/predictions tables on first run (SQLite file: backend/app.db)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Handwritten Math Symbol AI – Backend", version="0.2.0")

_default_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://sadik-2025.github.io",
    "https://handwritten-math-symbol-ai.onrender.com",
]
_env_origins = os.getenv("ALLOWED_ORIGINS")
allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(predict_router.router)
app.include_router(formula_router.router)
app.include_router(stats_router.router)
app.include_router(models_router.router)


# ---- Frontend ----
def _find_frontend_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "frontend",   # local: wex428-project/frontend
        here.parents[1] / "frontend",   # docker: /app/frontend
        Path.cwd() / "frontend",
        Path.cwd().parent / "frontend",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c
    raise RuntimeError("frontend folder not found. Checked: " + ", ".join(str(c) for c in candidates))


FRONTEND_DIR = _find_frontend_dir()

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")