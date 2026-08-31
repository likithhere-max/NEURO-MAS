from __future__ import annotations
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import agents, alerts, auth, chat, knowledge, ml_demo, patients, workflows
from app.core.config import get_settings
from app.db.session import Base, engine
from app.services.auth import get_current_user
settings = get_settings()
Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="NEURO-MAS API",
    description=(
        "A Multi-Agent Collaborative System for Autonomous and Personalized "
        "Neurological Healthcare. See /docs for the interactive API reference."
    ),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(patients.router, prefix=settings.api_v1_prefix)
app.include_router(workflows.router, prefix=settings.api_v1_prefix)
app.include_router(knowledge.router, prefix=settings.api_v1_prefix)
app.include_router(alerts.router, prefix=settings.api_v1_prefix)
app.include_router(agents.router, prefix=settings.api_v1_prefix)
app.include_router(ml_demo.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
@app.get("/")
def root():
    return {
        "system": "NEURO-MAS",
        "title": "A Multi-Agent Collaborative System for Autonomous and Personalized Neurological Healthcare",
        "docs": "/docs",
        "status": "ok",
    }
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get(f"{settings.api_v1_prefix}/whoami")
def whoami(user: dict = Depends(get_current_user)):
    return {"user_id": user["sub"], "email": user.get("email")}