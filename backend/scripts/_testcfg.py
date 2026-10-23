"""Shared guard + credentials for the LIVE integration scripts in this folder.

These scripts create users and rows in whatever Supabase project the environment points at. They therefore:
  * refuse to run unless NEUROMAS_TEST_PROJECT=yes is set (a deliberate statement that SUPABASE_URL is a throw-away
    TEST project, never the production one), and
  * read the test password from NEUROMAS_TEST_PASSWORD instead of a value committed to the repository.
"""
import os
import sys


def require_test_environment() -> None:
    if os.environ.get("NEUROMAS_TEST_PROJECT", "").lower() != "yes":
        sys.exit("Refusing to run: set NEUROMAS_TEST_PROJECT=yes to confirm SUPABASE_URL is a separate TEST project "
                 "(these scripts create users and rows).")


def test_password() -> str:
    require_test_environment()
    pw = os.environ.get("NEUROMAS_TEST_PASSWORD", "")
    if len(pw) < 12:
        sys.exit("Set NEUROMAS_TEST_PASSWORD (>= 12 characters) in the environment; no password is stored in the repo.")
    return pw
