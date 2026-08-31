from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChatMessageIn(BaseModel):
    message: str


class ChatMessageOut(BaseModel):
    id: Optional[str] = None
    role: str  # "user" | "assistant"
    content: str
    created_at: Optional[str] = None
