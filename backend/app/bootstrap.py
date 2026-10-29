import os

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from . import models
from .auth import hash_password
from .database import engine


def ensure_admin_column():
    cols = [c["name"] for c in inspect(engine).get_columns("users")]
    if "is_admin" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT FALSE"))


def ensure_admin_user():
    username = os.getenv("ADMIN_USERNAME")
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")
    if not (username and email and password):
        return
    with Session(engine) as db:
        user = db.query(models.User).filter(models.User.email == email).first()
        if user is None:
            db.add(models.User(
                username=username,
                email=email,
                hashed_password=hash_password(password),
                is_admin=True,
            ))
        else:
            user.is_admin = True
        db.commit()