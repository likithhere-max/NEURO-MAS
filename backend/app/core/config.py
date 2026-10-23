"""
Central configuration for NEURO-MAS.
Follows the "model-agnostic design" and "no hard-coded secrets" requirements:
all provider selection and secrets come from environment variables, never
from source code.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic import model_validator
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
    # Optional owner-supplied interaction-database snapshot (see app/services/ddi.py SnapshotSource). Empty = only the
    # illustrative curated table is used and the medication agent reports interactions as incompletely assessed.
    ddi_snapshot_path: str = ""
    # --- External (LLM) processing policy: see app/core/privacy.py ---
    # disabled   : nothing leaves the system; every narrative uses the deterministic template
    # redacted   : (default) best-effort masking of identifiers before sending
    # unredacted : send as-is (only for synthetic data / explicit owner decision)
    external_processing_mode: str = "redacted"
    # A patient-scoped external call needs patients.external_ai_consent = true (default false => fail closed).
    require_patient_consent: bool = True
    # --- Notifications (app/services/notification_dispatch.py) ---
    # "none" (default): nothing is ever sent and actions stay 'proposed'. "smtp": real e-mail via the settings below.
    notification_channel: str = "none"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_starttls: bool = True
    notification_max_attempts: int = 3
    notification_attempt_timeout_s: float = 10.0
    notification_backoff_base_s: float = 1.0
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
    # Demo/dev convenience ONLY: create accounts pre-confirmed (no email verification). Refused at startup unless
    # ENVIRONMENT=development. Default False: signup goes through Supabase's email-confirmation flow.
    auto_confirm_email: bool = False
    auth_rate_limit_per_minute: int = 10
    # Per-authenticated-user limits for expensive endpoints (0 disables).
    workflow_rate_limit_per_minute: int = 6
    chat_rate_limit_per_minute: int = 20
    # Number of trusted reverse proxies in front of the API (0 = ignore X-Forwarded-For). Wrong values let clients spoof IPs.
    trusted_proxy_count: int = 0
    # Optional shared store (redis://...). Empty = per-process in-memory limiter.
    rate_limit_redis_url: str = ""
    # --- CORS ---
    # "*" added 2026-09-02 as a same-day-review convenience: the frontend is
    # being hosted on Netlify (URL not known/fixed yet) and the backend is
    # being reached through a temporary ngrok tunnel, so the exact origin
    # can't be pinned in advance. Safe here because this API is
    # Bearer-token-authenticated (no cookies), never allow_credentials-reliant
    # in practice. Replace "*" with the real Netlify URL once it's stable and
    # remove this note.
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
    @model_validator(mode="after")
    def _refuse_unsafe_combinations(self):
        if self.auto_confirm_email and self.environment.lower() != "development":
            raise ValueError("AUTO_CONFIRM_EMAIL=true is only allowed when ENVIRONMENT=development "
                             "(it disables email verification)")
        if self.notification_channel.lower() == "smtp" and not (self.smtp_host and self.smtp_from):
            raise ValueError("NOTIFICATION_CHANNEL=smtp requires SMTP_HOST and SMTP_FROM")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()