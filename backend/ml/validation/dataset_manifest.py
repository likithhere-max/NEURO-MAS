"""
Versioned dataset manifest: verify files against it, and pin hashes (Phase 5, Stage 2).

verify_file() never returns a success-looking status for an unpinned manifest:
    PINNED_MATCH   file bytes equal the pinned SHA-256            (the only "good" outcome)
    UNPINNED       manifest records no hash yet                   (NOT a success)
    HASH_MISMATCH  file bytes differ from the pinned SHA-256      (refuse)
    MISSING_FILE   no file at the path
pin() refuses to record a hash unless the file (a) passes the structural verifier and (b) authenticates as the shipped training matrix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from ml.validation import data as D
from ml.validation import fingerprint as F

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "DATASET_MANIFEST.json"
KIND = {"parkinsons": "disease", "parkinsons_updrs": "risk"}

PINNED_MATCH, UNPINNED, HASH_MISMATCH, MISSING_FILE = "PINNED_MATCH", "UNPINNED", "HASH_MISMATCH", "MISSING_FILE"


class ManifestError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict:
    m = json.loads(Path(path).read_text())
    if m.get("manifest_schema") != "neuro-mas-dataset-manifest/1":
        raise ManifestError("unknown manifest schema")
    return m


def verify_file(name: str, csv: Path, manifest_path: Path = MANIFEST) -> dict:
    m = load_manifest(manifest_path)
    if name not in m["datasets"]:
        raise ManifestError(f"{name!r} is not in the manifest")
    csv = Path(csv)
    if not csv.is_file():
        return {"name": name, "status": MISSING_FILE, "sha256": None, "pinned_sha256": m["datasets"][name]["sha256"]}
    digest = D.sha256_file(csv)
    pinned = m["datasets"][name]["sha256"]
    status = UNPINNED if not pinned else (PINNED_MATCH if pinned == digest else HASH_MISMATCH)
    return {"name": name, "status": status, "sha256": digest, "pinned_sha256": pinned, "bytes": csv.stat().st_size}


def pin(name: str, csv: Path, manifest_path: Path = MANIFEST, models_dir: Path = F.MODELS, now: str | None = None) -> dict:
    """Record the file's SHA-256, but only for a file that is structurally valid AND authenticates as the shipped training matrix."""
    m = load_manifest(manifest_path)
    kind = KIND[name]
    csv = Path(csv)
    if not csv.is_file():
        raise ManifestError(f"{csv}: no such file")
    raw = pd.read_csv(csv)
    (D.verify_disease_frame if kind == "disease" else D.verify_risk_frame)(raw)         # raises DatasetVerificationError
    auth = F.authenticate(kind, csv, models_dir)
    if auth["status"] != F.AUTHENTICATED:
        raise ManifestError(f"refusing to pin {name}: the file does not authenticate as the shipped training data ({auth['status']})")
    digest = D.sha256_file(csv)
    prior = m["datasets"][name]["sha256"]
    if prior and prior != digest:
        raise ManifestError(f"{name} is already pinned to a different SHA-256; changing a pin is a protocol amendment")
    ds = m["datasets"][name]
    ds.update({"sha256": digest, "bytes": csv.stat().st_size, "pin_status": "PINNED_AND_AUTHENTICATED",
               "retrieved_utc": now or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    m["overall_status"] = "PINNED" if all(d["sha256"] for d in m["datasets"].values()) else "PARTIALLY_PINNED"
    Path(manifest_path).write_text(json.dumps(m, indent=2) + "\n")
    return ds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "verify", "pin"])
    ap.add_argument("--name", choices=sorted(KIND))
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "status":
            m = load_manifest()
            print(json.dumps({"overall_status": m["overall_status"], **{k: v["pin_status"] for k, v in m["datasets"].items()}}, indent=2))
            return 0 if m["overall_status"] == "PINNED" else 2
        if not a.name or not a.csv:
            ap.error("--name and --csv are required")
        if a.cmd == "verify":
            r = verify_file(a.name, Path(a.csv))
            print(json.dumps(r, indent=2))
            return 0 if r["status"] == PINNED_MATCH else 2
        print(json.dumps(pin(a.name, Path(a.csv)), indent=2))
        return 0
    except (ManifestError, D.DatasetVerificationError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
