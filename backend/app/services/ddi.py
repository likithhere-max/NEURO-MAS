"""
Drug-drug interaction (DDI) lookup with explicit, distinct outcomes (Phase 2, E1).

The bug class this removes: "no interaction in my table" silently reading as "safe". Every pair of the patient's
medications ends in exactly one of these statuses, and NONE of them means "safe":

  known_interaction               a checked source lists an interaction for the pair
  no_interaction_found_in_source  the source recognises both drugs, is exhaustive for pairs, and lists nothing
                                  (absence in ONE source; never proof of safety)
  drug_not_recognized             a source cannot identify at least one of the drugs, so cannot speak to the pair
  combination_not_assessed        both drugs are known to the source but it does not cover this pair (a curated,
                                  non-exhaustive table can only ever say this, never "no interaction")
  lookup_failed                   the source raised / timed out
  source_unavailable              no configured source is reachable / loadable
  source_outdated                 the source is older than its own max age; absence of a finding is not relied on
                                  (a *known* interaction from a stale source is still reported, flagged stale)

Sources implement `DDISource`. Shipped: `CuratedTableSource` (the 5-rule illustrative table; never exhaustive) and
`SnapshotSource` (an owner-supplied, versioned JSON export of a licensed/approved database). No live network source
is shipped: the NLM RxNav interaction API was reported discontinued in 2024 (verify), and no reachable replacement
could be tested from the build environment. Add one by subclassing `DDISource`.
"""
from __future__ import annotations

import itertools
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

MAX_DRUGS = 40
ABSENCE_STATEMENT = "A missing finding is not evidence that a combination is safe; clinician/pharmacist review is required."


class DDIStatus(str, Enum):
    KNOWN_INTERACTION = "known_interaction"
    NO_INTERACTION_FOUND_IN_SOURCE = "no_interaction_found_in_source"
    DRUG_NOT_RECOGNIZED = "drug_not_recognized"
    COMBINATION_NOT_ASSESSED = "combination_not_assessed"
    LOOKUP_FAILED = "lookup_failed"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_OUTDATED = "source_outdated"


# Which status wins when several sources answer for the same pair (most informative first).
_PRIORITY = [DDIStatus.KNOWN_INTERACTION, DDIStatus.NO_INTERACTION_FOUND_IN_SOURCE, DDIStatus.COMBINATION_NOT_ASSESSED,
             DDIStatus.DRUG_NOT_RECOGNIZED, DDIStatus.SOURCE_OUTDATED, DDIStatus.LOOKUP_FAILED, DDIStatus.SOURCE_UNAVAILABLE]
_DEFINITIVE = {DDIStatus.KNOWN_INTERACTION, DDIStatus.NO_INTERACTION_FOUND_IN_SOURCE}
_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}
_UNSPECIFIED_SEVERITY = {"", "unknown", "none", "unspecified"}


def severity_rank(label: Any) -> int:
    """Rank of a severity label for choosing the summary severity of a pair listed by several sources.

    Phase 3 (D04): the old key gave EVERY label outside {high, medium, low} (and any capitalised 'High') rank 0, so a source
    saying "contraindicated" lost to another source saying "low". Now: recognised labels (case-insensitive) keep their
    order; an unspecified severity carries no claim (rank 0); any OTHER non-empty label is an unrecognised claim and
    ranks above 'high' so the more alarming statement can never be hidden. The label itself is reported verbatim."""
    k = str(label or "").strip().lower()
    if k in _SEVERITY_ORDER:
        return _SEVERITY_ORDER[k]
    return 0 if k in _UNSPECIFIED_SEVERITY else 99


def normalize_name(name: Any) -> str:
    """Canonical key for a drug name: case-folded, accents removed, punctuation collapsed.

    Phase 3 (D05): the old key kept only [a-z0-9], so a name in another script (e.g. Devanagari) normalised to the EMPTY
    string and was silently dropped from the check, and 'L\u00e9vodopa' became 'l vodopa'. Latin accents are now folded
    ('levodopa'); a name that still contains non-Latin letters is kept in its own script (NFKC, case-folded) so it can be
    reported as unrecognised or matched against a snapshot that lists it."""
    raw = str(name or "")
    decomposed = unicodedata.normalize("NFKD", raw)
    # strip only the Combining Diacritical Marks block (U+0300-U+036F: Latin accents); Indic/other-script marks are kept
    folded = "".join(ch for ch in decomposed if not 0x300 <= ord(ch) <= 0x36F).casefold()
    if any(ch.isalpha() and not ch.isascii() for ch in folded):
        return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", raw).casefold()).strip()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


@dataclass
class PairAssessment:
    drug_a: str
    drug_b: str
    status: DDIStatus
    source_id: str
    source_version: str = ""
    retrieved_at: str | None = None
    severity: str | None = None
    description: str = ""
    reference: str | None = None
    stale: bool = False
    detail: str = ""


class DDISource(ABC):
    source_id: str = "abstract"
    version: str = ""
    retrieved_at: str | None = None            # ISO timestamp of the data snapshot
    max_age_days: float | None = None          # None = age is not enforced for this source
    exhaustive_pairs: bool = False             # True only if "not listed" really means "no known interaction"

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def is_outdated(self, now: datetime) -> bool:
        if self.max_age_days is None or not self.retrieved_at:
            return False
        try:
            ts = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
        except ValueError:
            return True                         # unparseable date -> cannot show it is fresh
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() > self.max_age_days * 86400

    @abstractmethod
    def recognizes(self, drug: str) -> bool: ...

    @abstractmethod
    def lookup(self, a: str, b: str) -> tuple[str, str, str | None] | None:
        """Return (severity, description, reference) if the pair interacts, else None. May raise."""


class CuratedTableSource(DDISource):
    """The illustrative curated table (clinical_rules.KNOWN_INTERACTIONS). Never exhaustive."""
    source_id = "curated_illustrative_table"
    exhaustive_pairs = False

    def __init__(self, rules: list[dict] | None = None):
        from app.services import clinical_rules as cr
        self._cr = cr
        self._rules = cr.KNOWN_INTERACTIONS if rules is None else rules
        self.version = f"curated-{len(self._rules)}-rules"
        self._covered = {r["drug_a"] for r in self._rules} | {r["drug_b"] for r in self._rules}

    def recognizes(self, drug: str) -> bool:
        return bool(self._cr.canonical_drug_names(drug) & self._covered)

    def lookup(self, a, b):
        ca, cb = self._cr.canonical_drug_names(a), self._cr.canonical_drug_names(b)
        for r in self._rules:
            if (r["drug_a"] in ca and r["drug_b"] in cb) or (r["drug_a"] in cb and r["drug_b"] in ca):
                return r["severity"], r["description"], r.get("source")
        return None


class SnapshotSource(DDISource):
    """Owner-supplied JSON export of an approved interaction database:

        {"source_id": "...", "version": "...", "retrieved_at": "2026-01-31T00:00:00Z", "license": "...",
         "max_age_days": 180, "exhaustive_pairs": true,
         "aliases": {"sinemet": "levodopa"}, "drugs": ["levodopa", ...],
         "interactions": [{"drug_a": "...", "drug_b": "...", "severity": "high", "description": "...", "reference": "..."}]}

    `exhaustive_pairs` must be true ONLY if the export genuinely lists every known interaction between the listed
    drugs; it is what permits the 'no_interaction_found_in_source' answer. A missing/invalid file makes the source
    unavailable (never silently empty).
    """

    def __init__(self, path: str | Path | None = None, data: dict | None = None):
        self._error: str | None = None
        self._drugs: set[str] = set()
        self._aliases: dict[str, str] = {}
        self._pairs: dict[frozenset, tuple[str, str, str | None]] = {}
        self.source_id = "snapshot"
        try:
            if data is None:
                if not path:
                    raise FileNotFoundError("no snapshot path configured")
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._load(data)
        except Exception as exc:               # unreadable or malformed snapshot -> unavailable, with a reason
            self._error = f"snapshot_unusable:{type(exc).__name__}"

    def _load(self, d: dict) -> None:
        if not isinstance(d, dict) or not isinstance(d.get("interactions"), list):
            raise ValueError("missing interactions list")
        self.source_id = str(d.get("source_id") or "snapshot")
        self.version = str(d.get("version") or "")
        self.retrieved_at = d.get("retrieved_at")
        self.max_age_days = d.get("max_age_days")
        self.exhaustive_pairs = bool(d.get("exhaustive_pairs", False))
        self._aliases = {normalize_name(k): normalize_name(v) for k, v in (d.get("aliases") or {}).items()}
        self._drugs = {normalize_name(x) for x in (d.get("drugs") or [])}
        for r in d["interactions"]:
            a, b = normalize_name(r["drug_a"]), normalize_name(r["drug_b"])
            self._drugs |= {a, b}
            self._pairs[frozenset((a, b))] = (str(r.get("severity") or "unknown"), str(r.get("description") or ""), r.get("reference"))

    def availability(self):
        return (self._error is None), self._error

    def _canon(self, drug: str) -> str:
        n = normalize_name(drug)
        return self._aliases.get(n, n)

    def recognizes(self, drug: str) -> bool:
        return self._canon(drug) in self._drugs

    def lookup(self, a, b):
        return self._pairs.get(frozenset((self._canon(a), self._canon(b))))


@dataclass
class PairResult:
    drug_a: str
    drug_b: str
    status: DDIStatus
    severity: str | None = None
    assessments: list[PairAssessment] = field(default_factory=list)


@dataclass
class DDIReport:
    drugs: list[str]
    pairs: list[PairResult]
    sources: list[dict]
    warnings: list[str]
    complete: bool
    unrecognized_drugs: list[str]
    statement: str = ABSENCE_STATEMENT

    @property
    def known(self) -> list[PairResult]:
        return [p for p in self.pairs if p.status == DDIStatus.KNOWN_INTERACTION]

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.pairs:
            out[p.status.value] = out.get(p.status.value, 0) + 1
        return out

    def as_dict(self) -> dict:
        return {
            "drugs": self.drugs, "complete": self.complete, "statement": self.statement, "warnings": self.warnings,
            "unrecognized_drugs": self.unrecognized_drugs, "sources": self.sources, "status_counts": self.status_counts(),
            "pairs": [{"drug_a": p.drug_a, "drug_b": p.drug_b, "status": p.status.value, "severity": p.severity,
                       "assessments": [{**a.__dict__, "status": a.status.value} for a in p.assessments]} for p in self.pairs],
        }


class DDIChecker:
    def __init__(self, sources: Iterable[DDISource]):
        self.sources = list(sources)

    def check(self, drug_names: Iterable[Any], now: datetime | None = None) -> DDIReport:
        now = now or datetime.now(timezone.utc)
        seen: dict[str, str] = {}
        unusable: list[str] = []
        for n in drug_names or []:
            k = normalize_name(n)
            if k and k not in seen:
                seen[k] = str(n).strip()
            elif not k and str(n or "").strip():
                unusable.append(str(n).strip())           # e.g. "???": cannot be checked -> reported, never dropped silently
        drugs = sorted(seen.values(), key=str.lower)
        warnings: list[str] = []
        if unusable:
            warnings.append(f"ddi_name_unusable: {unusable} could not be interpreted as drug names and were NOT checked")
        if len(drugs) > MAX_DRUGS:
            warnings.append(f"ddi_truncated: {len(drugs)} drugs supplied, only the first {MAX_DRUGS} were checked")
            drugs = drugs[:MAX_DRUGS]

        src_info, usable = [], []
        for s in self.sources:
            ok, why = s.availability()
            old = ok and s.is_outdated(now)
            src_info.append({"source_id": s.source_id, "version": s.version, "retrieved_at": s.retrieved_at,
                             "available": ok, "outdated": old, "exhaustive_pairs": s.exhaustive_pairs, "reason": why})
            if ok:
                usable.append((s, old))
            else:
                warnings.append(f"ddi_source_unavailable:{s.source_id}:{why}")
            if old:
                warnings.append(f"ddi_source_outdated:{s.source_id} (retrieved {s.retrieved_at}, max age {s.max_age_days} days)")
        if not usable:
            warnings.append("ddi_no_source_available: interactions could NOT be checked; this is not a clean result")

        pairs: list[PairResult] = []
        for a, b in itertools.combinations(drugs, 2):
            results: list[PairAssessment] = []
            if not usable:
                results.append(PairAssessment(a, b, DDIStatus.SOURCE_UNAVAILABLE, "none"))
            for s, old in usable:
                results.append(self._assess(s, old, a, b))
            pairs.append(self._combine(a, b, results))

        unrec = [d for d in drugs if usable and not any(s.recognizes(d) for s, _ in usable)]
        incomplete = [p for p in pairs if p.status not in _DEFINITIVE]
        if incomplete:
            counts: dict[str, int] = {}
            for p in incomplete:
                counts[p.status.value] = counts.get(p.status.value, 0) + 1
            warnings.append(f"ddi_incomplete: {len(incomplete)} of {len(pairs)} pair(s) not assessed {counts}")
        warnings.append("absence_of_a_finding_is_not_evidence_of_safety")
        return DDIReport(drugs=drugs, pairs=pairs, sources=src_info, warnings=warnings,
                         complete=bool(usable) and not incomplete and not unusable, unrecognized_drugs=unrec)

    @staticmethod
    def _assess(s: DDISource, old: bool, a: str, b: str) -> PairAssessment:
        base = dict(source_id=s.source_id, source_version=s.version, retrieved_at=s.retrieved_at)
        try:
            ra, rb = s.recognizes(a), s.recognizes(b)
            if not (ra and rb):
                missing = [d for d, r in ((a, ra), (b, rb)) if not r]
                return PairAssessment(a, b, DDIStatus.DRUG_NOT_RECOGNIZED, detail=f"not recognised: {missing}", **base)
            hit = s.lookup(a, b)
        except Exception as exc:               # a failing source must never read as 'no interaction'
            return PairAssessment(a, b, DDIStatus.LOOKUP_FAILED, detail=f"{type(exc).__name__}", **base)
        if hit is not None:
            sev, desc, ref = hit
            return PairAssessment(a, b, DDIStatus.KNOWN_INTERACTION, severity=sev, description=desc, reference=ref, stale=old, **base)
        if old:
            return PairAssessment(a, b, DDIStatus.SOURCE_OUTDATED, detail="source older than max age; absence not relied on", **base)
        if s.exhaustive_pairs:
            return PairAssessment(a, b, DDIStatus.NO_INTERACTION_FOUND_IN_SOURCE, **base)
        return PairAssessment(a, b, DDIStatus.COMBINATION_NOT_ASSESSED, detail="source is not exhaustive for pairs", **base)

    @staticmethod
    def _combine(a: str, b: str, results: list[PairAssessment]) -> PairResult:
        best = min(results, key=lambda r: _PRIORITY.index(r.status))
        sev = None
        if best.status == DDIStatus.KNOWN_INTERACTION:       # highest severity among all sources that list it
            sev = max((r.severity for r in results if r.status == DDIStatus.KNOWN_INTERACTION), key=severity_rank)
        return PairResult(a, b, best.status, sev, results)


_CACHE: dict[tuple, SnapshotSource] = {}


def get_default_checker() -> DDIChecker:
    """Curated illustrative table, plus the owner-supplied snapshot when DDI_SNAPSHOT_PATH is configured."""
    from app.core.config import get_settings
    sources: list[DDISource] = [CuratedTableSource()]
    path = getattr(get_settings(), "ddi_snapshot_path", "") or ""
    if path:
        try:
            key = (path, Path(path).stat().st_mtime_ns)
        except OSError:
            key = (path, None)
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = SnapshotSource(path)
        sources.append(_CACHE[key])
    return DDIChecker(sources)
