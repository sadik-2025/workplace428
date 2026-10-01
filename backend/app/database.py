import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# For dev, defaults to a local SQLite file (backend/app.db). Set DATABASE_URL
# env var to point at PostgreSQL/MySQL/etc for production, e.g.:
#   postgresql://user:password@localhost/wex428
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
