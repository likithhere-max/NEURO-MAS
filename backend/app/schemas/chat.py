from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatMessageOut(BaseModel):
    id: Optional[str] = None
    role: str  # "user" | "assistant"
    content: str
    created_at: Optional[str] = None
