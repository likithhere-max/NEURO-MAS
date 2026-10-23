"""Scan every COMMITTED blob reachable from any ref (all history) for credential-shaped strings. Uncommitted files are not scanned: commit first.
Never prints a matched value.

    python scripts/scan_secrets.py            # exit 1 if anything is found, 2 if the scan itself could not run

Method: enumerate (commit, path, blob) with `git rev-list` / `git ls-tree`, read each distinct blob once with
`git cat-file --batch`, and match with Python regexes (not `git grep`, whose regex dialect silently rejected the
Phase-2 patterns and made an earlier version of this script find nothing at all -- see tests/test_scan_secrets.py,
which plants secrets and requires them to be found).

Finding nothing is not proof of absence: the patterns cover common key shapes and quoted password literals only. A secret
that was ever committed must be treated as compromised and REVOKED at its provider; deleting it from files or history does
not revoke it. Rewriting shared history requires the repository owner's explicit decision; this script only reports."""
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
    # Low-entropy literals (e.g. a shared test-account password) have no recognisable shape; match assignments of a
    # quoted literal to a name containing "password"/"passwd"/"pwd". Environment lookups and empty values do not match.
    "password_literal": r"(?i)(?<!-)(?:pass(?:word|wd)|pwd)\w*[\"']?\s*[:=]\s*[\"'][^\"'\s${}]{6,}[\"']",
    "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
}
RX = {k: re.compile(v) for k, v in PATTERNS.items()}
# Known-benign fixtures (deliberately invalid tokens / placeholder URLs used by tests).
ALLOW_PATHS = ("backend/tests/", "tests/")
MAX_BLOB = 2_000_000


class ScanError(RuntimeError):
    pass


def git(*a, input_bytes=None):
    p = subprocess.run(["git", *a], capture_output=True, input=input_bytes)
    if p.returncode != 0:
        raise ScanError(f"git {' '.join(a[:2])} failed: {p.stderr.decode(errors='replace').strip()[:200]}")
    return p.stdout


def blobs():
    """yield (commit_short_or_WORKTREE, path, blob_sha) for every file in every commit reachable from any ref."""
    seen = set()
    commits = git("rev-list", "--all").decode().split()
    for c in commits:
        for line in git("ls-tree", "-r", "-z", c).decode(errors="replace").split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            _mode, typ, sha = meta.split()
            if typ == "blob" and (sha, path) not in seen:
                seen.add((sha, path))
                yield c[:8], path, sha


def read_blobs(shas):
    """{sha: bytes} using one `cat-file --batch` process."""
    out = {}
    uniq = list(dict.fromkeys(shas))
    if not uniq:
        return out
    p = subprocess.run(["git", "cat-file", "--batch"], capture_output=True, input=("\n".join(uniq) + "\n").encode())
    if p.returncode != 0:
        raise ScanError("git cat-file failed")
    buf, i = p.stdout, 0
    for sha in uniq:
        j = buf.index(b"\n", i)
        header = buf[i:j].split()
        if len(header) < 3:
            raise ScanError(f"unexpected cat-file header for {sha}")
        size = int(header[2])
        out[sha] = buf[j + 1: j + 1 + size]
        i = j + 1 + size + 1
    return out


def scan() -> list[tuple[str, str, str]]:
    git("rev-parse", "--git-dir")  # raises ScanError outside a repository
    items = [(c, p, s) for c, p, s in blobs() if not p.startswith(ALLOW_PATHS)]
    data = read_blobs([s for _, _, s in items])
    hits = set()
    for c, path, sha in items:
        b = data.get(sha, b"")
        if len(b) > MAX_BLOB or b"\0" in b[:4096]:
            continue
        text = b.decode("utf-8", errors="replace")
        for name, r in RX.items():
            if r.search(text):
                hits.add((c, path, name))
    return sorted(hits)


def main() -> int:
    try:
        hits = scan()
        n_commits = len(git("rev-list", "--all").decode().split())
    except ScanError as e:
        print(f"SCAN COULD NOT RUN: {e}", file=sys.stderr)
        return 2
    for c, path, name in hits:
        print(f"commit {c}  {path}  pattern={name}")
    print(f"{len(hits)} finding(s) across {n_commits} commit(s); values are never printed.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
