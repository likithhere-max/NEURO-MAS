"""
Row-Level-Security authorization matrix against a SEPARATE Supabase TEST project (Phase 2, F1).

STATUS: written and statically checked; NOT EXECUTED in the build environment (no test project / credentials were
available). Run it only against a throw-away project:

    export NEUROMAS_RLS_TEST_URL=https://<test-project>.supabase.co
    export NEUROMAS_RLS_TEST_PUBLISHABLE_KEY=...
    export NEUROMAS_RLS_TEST_SECRET_KEY=...
    export NEUROMAS_TEST_PROJECT=yes
    export NEUROMAS_TEST_PASSWORD='<12+ chars>'
    pytest tests/rls_live -v

Guards: skipped unless every variable is set AND the URL differs from the application's own SUPABASE_URL (so the
production project can never be targeted by mistake). It creates and deletes its own users/rows; it never touches
existing data. A PASS shows that THESE assertions hold on THAT project's policies, nothing more.
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.core.config import get_settings

URL = os.environ.get("NEUROMAS_RLS_TEST_URL", "")
PUB = os.environ.get("NEUROMAS_RLS_TEST_PUBLISHABLE_KEY", "")
SEC = os.environ.get("NEUROMAS_RLS_TEST_SECRET_KEY", "")
PW = os.environ.get("NEUROMAS_TEST_PASSWORD", "")


def harness_skip_reason(env: dict | None = None, app_url: str | None = None) -> str | None:
    """Pure function (unit-tested below, runs everywhere): why the live matrix must not run, or None."""
    e = env if env is not None else os.environ
    needed = ["NEUROMAS_RLS_TEST_URL", "NEUROMAS_RLS_TEST_PUBLISHABLE_KEY", "NEUROMAS_RLS_TEST_SECRET_KEY", "NEUROMAS_TEST_PASSWORD"]
    missing = [k for k in needed if not e.get(k)]
    if missing:
        return f"missing {missing}: no separate test project configured"
    if str(e.get("NEUROMAS_TEST_PROJECT", "")).lower() != "yes":
        return "NEUROMAS_TEST_PROJECT=yes not set"
    if len(e["NEUROMAS_TEST_PASSWORD"]) < 12:
        return "NEUROMAS_TEST_PASSWORD must be >= 12 characters"
    app = (app_url if app_url is not None else get_settings().supabase_url) or ""
    if app and e["NEUROMAS_RLS_TEST_URL"].rstrip("/") == app.rstrip("/"):
        return "REFUSING: test URL equals the application's SUPABASE_URL (would target the real project)"
    return None


# --------------------------------------------------------------------------- always-run harness self-tests
def test_guard_refuses_when_nothing_configured():
    assert "missing" in harness_skip_reason({}, "")


def test_guard_refuses_the_applications_own_project():
    env = {"NEUROMAS_RLS_TEST_URL": "https://abc.supabase.co/", "NEUROMAS_RLS_TEST_PUBLISHABLE_KEY": "k",
           "NEUROMAS_RLS_TEST_SECRET_KEY": "s", "NEUROMAS_TEST_PASSWORD": "x" * 12, "NEUROMAS_TEST_PROJECT": "yes"}
    assert "REFUSING" in harness_skip_reason(env, "https://abc.supabase.co")
    assert harness_skip_reason(env, "https://other.supabase.co") is None


def test_guard_requires_explicit_confirmation_and_strong_password():
    env = {"NEUROMAS_RLS_TEST_URL": "https://t.supabase.co", "NEUROMAS_RLS_TEST_PUBLISHABLE_KEY": "k",
           "NEUROMAS_RLS_TEST_SECRET_KEY": "s", "NEUROMAS_TEST_PASSWORD": "short", "NEUROMAS_TEST_PROJECT": "yes"}
    assert "12" in harness_skip_reason(env, "")
    env.update(NEUROMAS_TEST_PASSWORD="x" * 12, NEUROMAS_TEST_PROJECT="no")
    assert "NEUROMAS_TEST_PROJECT" in harness_skip_reason(env, "")


# --------------------------------------------------------------------------- live matrix (skipped by default)
SKIP = harness_skip_reason()
live = pytest.mark.skipif(SKIP is not None, reason=SKIP or "")

PATIENT_SCOPED_TABLES = ["conditions", "allergies", "medications", "longitudinal_metrics", "metric_readings",
                         "observations", "workflows", "agent_event_log", "alerts", "actions", "audit_events",
                         "symptom_assessments", "chat_messages"]


@pytest.fixture(scope="module")
def world():
    from supabase import create_client
    admin = create_client(URL, SEC)
    tag = uuid.uuid4().hex[:8]
    users, ids, created_patients = {}, {}, []

    def mk(role):
        email = f"rls.{role}.{tag}@example.com"
        u = admin.auth.admin.create_user({"email": email, "password": PW, "email_confirm": True}).user
        ids[role] = u.id
        pub = create_client(URL, PUB)
        tok = pub.auth.sign_in_with_password({"email": email, "password": PW}).session.access_token
        c = create_client(URL, PUB)
        c.postgrest.auth(tok)
        c.table("profiles").insert({"id": u.id, "full_name": role, "role": "clinician" if role.startswith("clin") else "patient"}).execute()
        users[role] = c

    for r in ("clin_a", "clin_b", "pat_a", "stranger"):
        mk(r)
    pa = users["clin_a"].table("patients").insert({"full_name": f"RLS-A-{tag}", "clinician_id": ids["clin_a"]}).execute().data[0]
    pb = users["clin_b"].table("patients").insert({"full_name": f"RLS-B-{tag}", "clinician_id": ids["clin_b"]}).execute().data[0]
    created_patients += [pa["id"], pb["id"]]
    admin.table("conditions").insert({"patient_id": pb["id"], "preferred_name": "rls-secret-condition"}).execute()
    yield {"admin": admin, "users": users, "ids": ids, "pa": pa, "pb": pb, "anon": create_client(URL, PUB)}
    for pid in created_patients:                       # cleanup: only rows this run created
        for t in PATIENT_SCOPED_TABLES:
            try:
                admin.table(t).delete().eq("patient_id", pid).execute()
            except Exception:
                pass
        admin.table("patients").delete().eq("id", pid).execute()
    for uid in ids.values():
        try:
            admin.table("profiles").delete().eq("id", uid).execute()
            admin.auth.admin.delete_user(uid)
        except Exception:
            pass


@live
def test_clinician_reads_own_patient_but_not_anothers(world):
    a = world["users"]["clin_a"]
    assert [r["id"] for r in a.table("patients").select("id").eq("id", world["pa"]["id"]).execute().data] == [world["pa"]["id"]]
    assert a.table("patients").select("id").eq("id", world["pb"]["id"]).execute().data == []


@live
def test_clinician_cannot_update_or_delete_another_clinicians_patient(world):
    a = world["users"]["clin_a"]
    assert a.table("patients").update({"full_name": "HACKED"}).eq("id", world["pb"]["id"]).execute().data == []
    a.table("patients").delete().eq("id", world["pb"]["id"]).execute()
    still = world["admin"].table("patients").select("full_name").eq("id", world["pb"]["id"]).execute().data
    assert still and still[0]["full_name"] != "HACKED"


@live
def test_child_tables_hide_other_patients_rows(world):
    a = world["users"]["clin_a"]
    assert a.table("conditions").select("*").eq("patient_id", world["pb"]["id"]).execute().data == []


@live
def test_cannot_insert_child_rows_for_a_patient_you_do_not_own(world):
    a = world["users"]["clin_a"]
    with pytest.raises(Exception):
        a.table("conditions").insert({"patient_id": world["pb"]["id"], "preferred_name": "planted"}).execute()
    planted = world["admin"].table("conditions").select("*").eq("preferred_name", "planted").execute().data
    assert planted == []


@live
def test_unrelated_authenticated_user_sees_no_patients(world):
    assert world["users"]["stranger"].table("patients").select("id").execute().data == []


@live
@pytest.mark.parametrize("table", ["patients"] + PATIENT_SCOPED_TABLES)
def test_anonymous_requests_get_nothing(world, table):
    try:
        rows = world["anon"].table(table).select("*").limit(5).execute().data
    except Exception:
        return                                         # an explicit permission error is also acceptable
    assert rows == []


@live
def test_user_cannot_escalate_own_role(world):
    s = world["users"]["stranger"]
    try:
        s.table("profiles").update({"role": "clinician"}).eq("id", world["ids"]["stranger"]).execute()
    except Exception:
        pass
    role = world["admin"].table("profiles").select("role").eq("id", world["ids"]["stranger"]).execute().data[0]["role"]
    assert role == "patient"


@live
def test_user_cannot_read_other_profiles(world):
    s = world["users"]["stranger"]
    assert s.table("profiles").select("id").eq("id", world["ids"]["clin_a"]).execute().data == []


@live
def test_consent_flag_is_not_writable_by_unrelated_users(world):
    s = world["users"]["stranger"]
    try:
        s.table("patients").update({"external_ai_consent": True}).eq("id", world["pa"]["id"]).execute()
    except Exception:
        pass
    v = world["admin"].table("patients").select("external_ai_consent").eq("id", world["pa"]["id"]).execute().data[0]
    assert v.get("external_ai_consent") in (False, None)
