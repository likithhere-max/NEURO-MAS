"""
Step 7 test script -- creates a SECOND test user and confirms they cannot
read or modify the first test user's profile row. This is the deny-side
proof that Step 6's allow-side proof was missing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.supabase_client import get_supabase_admin_client, get_supabase_public_client

USER_1_ID = "7cea8de3-e08b-4b07-8b21-690e82b392a8"  # from Step 6's output

TEST_EMAIL_2 = "neuromas.test.user2@example.com"
TEST_PASSWORD_2 = "TestPassword123!"

admin = get_supabase_admin_client()
public = get_supabase_public_client()

created = admin.auth.admin.create_user({
    "email": TEST_EMAIL_2,
    "password": TEST_PASSWORD_2,
    "email_confirm": True,
})
user_2_id = created.user.id
print(f"Created second test user: {user_2_id}")

public.auth.sign_in_with_password({"email": TEST_EMAIL_2, "password": TEST_PASSWORD_2})
print("Signed in as user 2.")

# Try to read user 1's row -- should come back EMPTY, not an error. RLS
# filters it out silently, as if the row doesn't exist for this user.
leak_attempt = public.table("profiles").select("*").eq("id", USER_1_ID).execute()
print("Attempt to read user 1's row as user 2:", leak_attempt.data)

# Try to update user 1's row -- should also affect 0 rows.
tamper_attempt = public.table("profiles").update({"full_name": "HACKED"}).eq("id", USER_1_ID).execute()
print("Attempt to update user 1's row as user 2:", tamper_attempt.data)

if leak_attempt.data == [] and tamper_attempt.data == []:
    print("PASS: RLS correctly denied cross-user access.")
else:
    print("FAIL: RLS did NOT block cross-user access -- stop and check your policies.")