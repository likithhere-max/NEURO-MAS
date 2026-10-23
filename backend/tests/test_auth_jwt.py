"""JWT verification: signature, audience, issuer, expiry, required claims (audit: issuer was not checked)."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import app.services.auth as auth
from app.core.config import Settings

KEY = ec.generate_private_key(ec.SECP256R1())
OTHER = ec.generate_private_key(ec.SECP256R1())
SUPA = "https://proj.supabase.co"


class _SigningKey:
    def __init__(self, key): self.key = key.public_key()


class _JWKS:
    def get_signing_key_from_jwt(self, token): return _SigningKey(KEY)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(auth, "get_jwks_client", lambda: _JWKS())
    monkeypatch.setattr(auth, "get_settings", lambda: Settings(_env_file=None, supabase_url=SUPA))


def _token(key=KEY, **over):
    claims = {"sub": "user-1", "aud": "authenticated", "iss": f"{SUPA}/auth/v1", "exp": int(time.time()) + 600, "email": "a@b.co"}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="ES256")


def _verify(tok):
    return auth.get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok))


def test_valid_token_accepted():
    u = _verify(_token())
    assert u["sub"] == "user-1" and u["_access_token"]


@pytest.mark.parametrize("tok", [
    _token(iss="https://evil.example/auth/v1"),     # validly signed by the same key, wrong issuer
    _token(aud="anon"),
    _token(exp=int(time.time()) - 10),
    _token(sub=None),
    _token(key=OTHER),                               # wrong signing key
    "not.a.jwt",
])
def test_bad_tokens_rejected(tok):
    with pytest.raises(HTTPException) as e:
        _verify(tok)
    assert e.value.status_code == 401


def test_hs256_alg_confusion_rejected():
    tok = jwt.encode({"sub": "x", "aud": "authenticated", "iss": f"{SUPA}/auth/v1", "exp": int(time.time()) + 60}, "secret" * 8, algorithm="HS256")
    with pytest.raises(HTTPException):
        _verify(tok)
