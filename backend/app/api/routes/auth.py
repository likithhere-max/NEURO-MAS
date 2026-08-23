from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from app.core.supabase_client import get_supabase_admin_client, get_supabase_public_client
from app.schemas.api import Token, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup", response_model=Token)
def signup(payload: UserCreate):
    if payload.role not in {"clinician", "patient", "caregiver"}:
        raise HTTPException(400, "role must be one of clinician, patient, caregiver")

    admin = get_supabase_admin_client()
    public = get_supabase_public_client()

    try:
        created = admin.auth.admin.create_user({
            "email": payload.email,
            "password": payload.password,
            "email_confirm": True,  # prototype simplification -- see docs/ROADMAP.md
        })
    except Exception as exc:
        raise HTTPException(400, f"Could not create account: {exc}")

    user_id = created.user.id

    # Sign in as the brand-new user immediately, so the profile insert below
    # runs AS THEM -- respecting Row Level Security -- instead of using the
    # admin key to force it through.
    try:
        session = public.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password,
        })
    except Exception as exc:
        raise HTTPException(500, f"Account created but sign-in failed: {exc}")

    public.table("profiles").insert({
        "id": user_id,
        "full_name": payload.full_name,
        "role": payload.role,
    }).execute()

    return Token(access_token=session.session.access_token, role=payload.role)


@router.post("/login", response_model=Token)
def login(payload: LoginRequest):
    public = get_supabase_public_client()

    try:
        session = public.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password,
        })
    except Exception:
        raise HTTPException(401, "invalid credentials")

    profile = (
        public.table("profiles")
        .select("role")
        .eq("id", session.user.id)
        .single()
        .execute()
    )
    role = profile.data["role"] if profile.data else "patient"

    return Token(access_token=session.session.access_token, role=role)