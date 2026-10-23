"""
Step 6 test script -- creates one real test user, signs in as them, and
proves Row Level Security actually enforces "you can only touch your own
row" using a genuine signed-in session (not the admin key, which bypasses
RLS entirely).

Safe-ish to re-run, but Supabase will reject creating the same email twice.
If you need to re-run this, delete the test user first in the Supabase
dashboard: Authentication -> Users.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _testcfg  # live-script guard + env-provided password
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.supabase_client import get_supabase_admin_client, get_supabase_public_client

TEST_EMAIL = "neuromas.test.user@example.com"
TEST_PASSWORD = _testcfg.test_password()

admin = get_supabase_admin_client()
public = get_supabase_public_client()

created = admin.auth.admin.create_user({
    "email": TEST_EMAIL,
    "password": TEST_PASSWORD,
    "email_confirm": True,
})
user_id = created.user.id
print(f"Created test user: {user_id}")

session = public.auth.sign_in_with_password({"email": TEST_EMAIL, "password": TEST_PASSWORD})
print("Signed in. Access token acquired:", bool(session.session.access_token))

insert_result = public.table("profiles").insert({
    "id": user_id,
    "full_name": "Test Patient",
    "role": "patient",
}).execute()
print("Insert result:", insert_result.data)

select_result = public.table("profiles").select("*").execute()
print("Select result (should show exactly 1 row, their own):", select_result.data)