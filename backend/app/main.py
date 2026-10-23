from __future__ import annotations
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import agents, alerts, auth, chat, knowledge, ml_demo, patients, workflows
from app.core.config import get_settings
from app.services.auth import get_current_user
settings = get_settings()
app = FastAPI(
    title="NEURO-MAS API",
    description=(
        "A Multi-Agent Collaborative System for Autonomous and Personalized "
        "Neurological Healthcare. See /docs for the interactive API reference."
    ),
    version="0.1.0",
)
# Audit fix: "*" origins combined with allow_credentials=True. This API authenticates with a Bearer
# header (no cookies), so credentials mode is not needed; it is disabled whenever "*" is present.
_wildcard_cors = "*" in settings.cors_origins
if _wildcard_cors and settings.environment.lower() != "development":
    logging.getLogger("neuromas").warning("CORS allows any origin (\"*\") in a non-development environment; pin CORS_ORIGINS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not _wildcard_cors,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    """Audit fix: FastAPI's default handler JSON-encodes the rejected `input`, which (a) crashes with a 500
    when the input is NaN/Infinity ("Out of range float values are not JSON compliant") and (b) echoes
    patient-supplied text back in error bodies. Return only location, message and type."""
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


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