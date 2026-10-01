from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import auth_router, formula_router, models_router, predict_router, stats_router

# Creates users/predictions tables on first run (SQLite file: backend/app.db)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Handwritten Math Symbol AI — Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin before deploying
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
