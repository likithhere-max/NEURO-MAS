"""Positive controls for scripts/scan_secrets.py: a scanner that cannot find a planted secret proves nothing."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_secrets.py"


def _repo(tmp_path, content: str, name="config.py"):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com"); run("config", "user.name", "t")
    (tmp_path / name).write_text(content)
    run("add", "-A"); run("commit", "-qm", "c")
    return tmp_path


def _scan(cwd):
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=cwd, capture_output=True, text=True)


@pytest.mark.parametrize("planted,pattern", [
    ('API = "AIza' + "A" * 35 + '"\n', "google_api_key"),
    ('token = "ghp_' + "a" * 30 + '"\n', "github_token"),
    ('DB = "postgresql://user:hunter2pw@db.example.com:5432/x"\n', "db_url_with_password"),
    ('PASSWORD = "Sup3rSecretPw"\n', "password_literal"),
    ('TEST_PASSWORD_2 = "Sup3rSecretPw"\n', "password_literal"),            # suffixed names
    ('creds = {"password": "Sup3rSecretPw"}\n', "password_literal"),        # dict key form
    ('x = "-----BEGIN RSA PRIVATE KEY-----"\n', "private_key_block"),
])
def test_planted_secret_is_found_and_value_is_never_printed(tmp_path, planted, pattern):
    r = _scan(_repo(tmp_path, planted))
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"pattern={pattern}" in r.stdout
    for leaked in ("Sup3rSecretPw", "hunter2pw", "ghp_" + "a" * 30, "AIza" + "A" * 35):
        assert leaked not in r.stdout + r.stderr


def test_clean_repo_and_env_lookups_are_not_flagged(tmp_path):
    r = _scan(_repo(tmp_path, 'import os\npassword = os.environ["PW"]\nPASSWORD = ""\nx = 1\n'))
    assert r.returncode == 0, r.stdout + r.stderr


def test_html_autocomplete_values_are_not_flagged(tmp_path):
    r = _scan(_repo(tmp_path, 'x = mode === "login" ? "current-password" : "new-password"\n', name="page.tsx"))
    assert r.returncode == 0, r.stdout


def test_secret_removed_from_tree_is_still_found_in_history(tmp_path):
    repo = _repo(tmp_path, 'PASSWORD = "Sup3rSecretPw"\n')
    (repo / "config.py").write_text('PASSWORD = ""\n')
    subprocess.run(["git", "commit", "-qam", "remove"], cwd=repo, check=True, capture_output=True)
    assert _scan(repo).returncode == 1


def test_scanner_fails_loudly_if_git_grep_errors(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True, text=True)  # not a git repo
    assert r.returncode == 2
