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

    # --- Auth ---
    secret_key: str = "CHANGE_ME_INSECURE_DEV_ONLY_SECRET_KEY"
    access_token_expire_minutes: int = 60 * 8
    jwt_algorithm: str = "HS256"

    # --- LLM / model providers (model-agnostic; see app/services/llm_provider.py) ---
    # If no key is configured, the system falls back to a deterministic,
    # template-based reasoning provider and clearly labels output as such.
    # It NEVER silently fabricates a "smarter" answer.
    llm_provider: str = "template"  # "template" | "anthropic" | "openai"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Safety / consensus tuning (explicitly NOT claimed to be clinically validated) ---
    consensus_disagreement_threshold: float = 0.4
    min_confidence_for_autonomous_action: float = 0.6

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()