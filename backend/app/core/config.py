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
    # Per-agent wall-clock limit (seconds). An agent that exceeds it is recorded as FAILED,
    # never allowed to stall the emergency path indefinitely.
    agent_timeout_seconds: float = 60.0
    # Whole-workflow wall-clock budget (seconds). Each agent's timeout is min(agent_timeout_seconds, time left);
    # agents that cannot finish inside the budget are recorded TIMED_OUT and the workflow returns a structured
    # partial result instead of hanging.
    workflow_deadline_seconds: float = 120.0
    # Persist a finished workflow through ONE database transaction (RPC persist_workflow_atomic, see
    # scripts/migration_phase2_persistence.sql). False = sequential inserts with an explicit persist_state marker.
    # Only turn on after the migration has been applied to the target project.
    persist_via_rpc: bool = False
    # Max agents executing at the same time (bounds LLM/CPU fan-out).
    max_agent_concurrency: int = 4
    # Hard limits on patient-supplied free text (abuse / cost / prompt-injection surface).
    max_free_text_chars: int = 5000
    max_chat_message_chars: int = 2000
    # --- Account security ---
    # Clinicians can approve restricted actions, so self-registering as one must not be open to
    # the internet. If CLINICIAN_INVITE_CODE is set, clinician signup requires it. If it is NOT
    # set, clinician self-signup is allowed only when ENVIRONMENT=development (local demos).
    clinician_invite_code: str = ""
    auth_rate_limit_per_minute: int = 10
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