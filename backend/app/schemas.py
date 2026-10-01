from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, EmailStr, ConfigDict


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictionOut(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    top_k: List[Dict]
    class_probabilities: Dict[str, float]
    model_used: str
    created_at: datetime
