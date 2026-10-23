from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.core.config import get_settings
from app.core.ratelimit import rate_limit_auth
from app.core.supabase_client import get_authed_client, get_supabase_admin_client, new_public_client
from app.schemas.api import Token, UserCreate

logger = logging.getLogger("neuromas.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


def _check_clinician_signup_allowed(payload: UserCreate) -> None:
    """Audit fix (Critical): anyone on the internet could self-register as `clinician` and then
    approve restricted actions. Clinician signup now needs the configured invite code; with no
    code configured it is allowed only in a development environment."""
    settings = get_settings()
    if settings.clinician_invite_code:
        if not payload.invite_code or not hmac.compare_digest(payload.invite_code, settings.clinician_invite_code):
            raise HTTPException(403, "a valid invite code is required to register as a clinician")
    elif settings.environment.lower() != "development":
        raise HTTPException(403, "clinician self-registration is disabled")


@router.post("/signup", response_model=Token, dependencies=[Depends(rate_limit_auth)])
def signup(payload: UserCreate):
    if payload.role not in {"clinician", "patient", "caregiver"}:
        raise HTTPException(400, "role must be one of clinician, patient, caregiver")
    if payload.role == "clinician":
        _check_clinician_signup_allowed(payload)

    admin = get_supabase_admin_client()
    public = new_public_client()  # per-request: sign-in state must not be shared across requests

    try:
        created = admin.auth.admin.create_user({
            "email": payload.email,
            "password": payload.password,
            "email_confirm": True,  # KNOWN GAP: no email verification (needs SMTP configured in Supabase)
        })
    except Exception as exc:
        logger.warning("signup_create_user_failed error=%r", exc)
        raise HTTPException(400, "could not create account (the email may already be registered)")

    user_id = created.user.id

    # Sign in as the brand-new user so the profile insert runs AS THEM (RLS-respecting).
    try:
        session = public.auth.sign_in_with_password({"email": payload.email, "password": payload.password})
        get_authed_client(session.session.access_token).table("profiles").insert({
            "id": user_id, "full_name": payload.full_name, "role": payload.role,
        }).execute()
    except Exception as exc:
        # Do not leave an orphan auth user without a profile (login would break for it).
        logger.error("signup_profile_step_failed user_id=%s error=%r", user_id, exc)
        try:
            admin.auth.admin.delete_user(user_id)
        except Exception as cleanup_exc:  # pragma: no cover
            logger.error("signup_cleanup_failed user_id=%s error=%r", user_id, cleanup_exc)
        raise HTTPException(500, "account setup failed; please try again")

    return Token(access_token=session.session.access_token, role=payload.role)


@router.post("/login", response_model=Token, dependencies=[Depends(rate_limit_auth)])
def login(payload: LoginRequest):
    public = new_public_client()

    try:
        session = public.auth.sign_in_with_password({"email": payload.email, "password": payload.password})
    except Exception:
        raise HTTPException(401, "invalid credentials")

    profile = get_authed_client(session.session.access_token).table("profiles").select("role").eq("id", session.user.id).execute()
    if not profile.data:
        raise HTTPException(403, "account has no profile; contact an administrator")
    return Token(access_token=session.session.access_token, role=profile.data[0]["role"])
