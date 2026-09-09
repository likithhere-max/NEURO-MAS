"""
Central configuration for NEURO-MAS.
Follows the "model-agnostic design" and "no hard-coded secrets" requirements:
all provider selection and secrets come from environment variables, never
from source code.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # --- App ---
    app_name: str = "NEURO-MAS"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    # --- Database ---
    # Defaults to a local SQLite file so the project runs with zero external
    # infrastructure for academic/demo purposes. Point DATABASE_URL at a
    # Postgres instance (e.g. postgresql+psycopg://...) for production use --
    # the SQLAlchemy models are written to be portable across both.
    database_url: str = "sqlite:///./neuromas.db"
    # --- Supabase ---
    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_secret_key: str = ""
    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_default_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimensions: int = 768
    # NOTE (Roadmap item 7 housekeeping): the old local-JWT "--- Auth ---"
    # block (secret_key / jwt_algorithm / access_token_expire_minutes) was
    # removed here -- verified unused anywhere in the codebase. Auth is
    # handled entirely via Supabase JWT verification (JWKS/ES256) in
    # app/services/auth.py now, which needs no local signing secret.
    # --- LLM / model providers (model-agnostic; see app/services/llm_provider.py) ---
    # If no key is configured, the system falls back to a deterministic,
    # template-based reasoning provider and clearly labels output as such.
    # It NEVER silently fabricates a "smarter" answer.
    # NOTE (2026-08-24): in practice get_llm_provider() always returns
    # GeminiLLMProvider since gemini_api_key is always set -- the
    # llm_provider/anthropic_api_key branches below are vestigial but kept
    # working intentionally (Roadmap item 7 housekeeping decision: scoped to
    # removing only the fully-dead openai_api_key field, not retiring this
    # whole abstraction). Revisit later if desired.
    llm_provider: str = "template"  # "template" | "anthropic"
    anthropic_api_key: str | None = None
    # --- Safety / consensus tuning (explicitly NOT claimed to be clinically validated) ---
    consensus_disagreement_threshold: float = 0.4
    min_confidence_for_autonomous_action: float = 0.6
    # --- CORS ---
    # "*" added 2026-09-02 as a same-day-review convenience: the frontend is
    # being hosted on Netlify (URL not known/fixed yet) and the backend is
    # being reached through a temporary ngrok tunnel, so the exact origin
    # can't be pinned in advance. Safe here because this API is
    # Bearer-token-authenticated (no cookies), never allow_credentials-reliant
    # in practice. Replace "*" with the real Netlify URL once it's stable and
    # remove this note.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
@lru_cache
def get_settings() -> Settings:
    return Settings()