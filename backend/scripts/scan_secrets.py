"""Scan the working tree and ALL git history for credential-shaped strings. Never prints a matched value.

    python scripts/scan_secrets.py            # exit 1 if anything is found

Finding nothing is not proof of absence (regexes cover common key shapes only), and a secret that was ever committed
must be treated as compromised and REVOKED at its provider: deleting it from files or history does not revoke it.
Rewriting shared git history requires the repository owner's explicit decision; this script only reports."""
import re
import subprocess
import sys

PATTERNS = {
    "jwt_like": r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "supabase_secret_key": r"sb_secret_[A-Za-z0-9_-]{10,}",
    "google_api_key": r"AIza[0-9A-Za-z_-]{30,}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "generic_sk_key": r"\bsk-[A-Za-z0-9]{20,}",
    "db_url_with_password": r"postgres(?:ql)?://[^\s\"'/:]+:[^\s\"'@]{3,}@",
    "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
}
# Known-benign fixtures (deliberately invalid tokens / placeholder URLs used by tests).
ALLOW_PATHS = ("backend/tests/",)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, errors="replace").stdout


def main() -> int:
    hits = []
    rx = {k: re.compile(v) for k, v in PATTERNS.items()}
    for commit in git("rev-list", "--all").split():
        for line in git("grep", "-IE", "|".join(PATTERNS.values()), commit, "--", ".").splitlines():
            path = line.split(":", 2)[1] if line.count(":") >= 2 else line
            if path.startswith(ALLOW_PATHS):
                continue
            for name, r in rx.items():
                if r.search(line):
                    hits.append((commit[:8], path, name))
    for h in sorted(set(hits)):
        print(f"commit {h[0]}  {h[1]}  pattern={h[2]}")
    print(f"{len(set(hits))} finding(s) across {len(git('rev-list', '--all').split())} commit(s); values are never printed.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
