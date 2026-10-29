import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from . import models
from .auth import hash_password, require_admin
from .database import Base, engine, get_db
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


# ---- Admin ----
class AdminUserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    is_admin: bool = False


def _user_dict(u: models.User) -> dict:
    return {"id": u.id, "username": u.username, "email": u.email, "is_admin": bool(u.is_admin)}


@app.get("/admin/users", tags=["admin"])
def list_users(admin=Depends(require_admin), db: Session = Depends(get_db)):
    return [_user_dict(u) for u in db.query(models.User).order_by(models.User.id).all()]


@app.post("/admin/users", tags=["admin"], status_code=201)
def create_user(data: AdminUserCreate, admin=Depends(require_admin), db: Session = Depends(get_db)):
    username = data.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    exists = (
        db.query(models.User)
        .filter((models.User.username == username) | (models.User.email == data.email))
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    user = models.User(
        username=username,
        email=data.email,
        hashed_password=hash_password(data.password),
        is_admin=data.is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_dict(user)


@app.delete("/admin/users/{user_id}", tags=["admin"])
def delete_user(user_id: int, admin=Depends(require_admin), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.query(models.Prediction).filter(models.Prediction.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"deleted": user_id}


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