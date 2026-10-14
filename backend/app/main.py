import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import auth_router, formula_router, models_router, predict_router, stats_router

# Creates users/predictions tables on first run (SQLite file: backend/app.db)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Handwritten Math Symbol AI — Backend", version="0.2.0")

# Set ALLOWED_ORIGINS env var (comma-separated) on Render to override, e.g.:
#   ALLOWED_ORIGINS=https://sadik-2025.github.io
# Falls back to localhost + the known GitHub Pages origin for local/dev use.
_default_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://sadik-2025.github.io",
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


@app.get("/")
def root():
    return {"message": "Handwritten Math Symbol AI backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}