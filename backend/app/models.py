from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_used = Column(String, nullable=False)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    class_probabilities = Column(Text, nullable=True)  # JSON-encoded string
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")
