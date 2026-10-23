"""Configurable, provenance-carrying change thresholds (Phase 2, C). MCID applies ONLY when clinician-approved."""
import json

import pytest

from app.core import metric_semantics as ms
from app.core.config import Settings


def _use(monkeypatch, tmp_path=None, mcid=None, **kw):
    if mcid is not None:
        p = tmp_path / "mcid.json"
        p.write_text(json.dumps(mcid))
        kw["metric_mcid_path"] = str(p)
    s = Settings(_env_file=None, **kw)
    monkeypatch.setattr(ms, "get_settings", lambda: s)
    ms._load_mcid.cache_clear()


def test_default_thresholds_are_labelled_provisional(monkeypatch):
    _use(monkeypatch)
    r = ms.classify_change("gait_speed", 1.2, 0.7)
    assert r["classification"] == "deteriorating"
    assert r["threshold_basis"] == "provisional_relative_change" and r["threshold_provisional"] is True


def test_adverse_threshold_is_configurable(monkeypatch):
    _use(monkeypatch, metric_adverse_threshold=0.5)
    assert ms.classify_change("updrs_total", 20, 28)["classification"] == "stable"        # +40% < 50%
    assert ms.classify_change("updrs_total", 20, 31)["classification"] == "deteriorating"  # +55%


def test_invalid_thresholds_are_refused_at_startup():
    with pytest.raises(ValueError):
        Settings(_env_file=None, metric_stable_band=0.3, metric_adverse_threshold=0.2)
    with pytest.raises(ValueError):
        Settings(_env_file=None, metric_stable_band=-0.1)


def test_unapproved_mcid_is_ignored_with_a_note(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, mcid={"updrs_total": {"mcid": 3.0}})          # no approver / source
    r = ms.classify_change("updrs_total", 20, 24)                              # +20% -> provisional rule applies
    assert r["threshold_basis"] == "provisional_relative_change"
    assert "mcid_ignored_not_clinician_approved" in r["note"]


def test_approved_mcid_overrides_relative_rule_and_carries_provenance(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, mcid={"updrs_total": {"mcid": 3.0, "approved_by": "Dr. Example", "source": "local protocol v2"}})
    worse = ms.classify_change("updrs_total", 40, 43.5)    # +8.75% relative (would be 'stable' provisionally), but >= MCID
    assert worse["classification"] == "deteriorating"
    assert worse["threshold_basis"] == "clinician_approved_mcid" and worse["threshold_provisional"] is False
    assert worse["threshold_source"] == "local protocol v2"
    assert ms.classify_change("updrs_total", 40, 38.0)["classification"] == "stable"       # |change| < MCID
    assert ms.classify_change("updrs_total", 40, 36.0)["classification"] == "improving"


def test_mcid_respects_polarity(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, mcid={"moca": {"mcid": 2.0, "approved_by": "x", "source": "y"}})
    assert ms.classify_change("moca", 26, 23)["classification"] == "deteriorating"   # lower is worse
    assert ms.classify_change("moca", 26, 29)["classification"] == "improving"


def test_malformed_mcid_file_fails_closed_to_provisional(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"; p.write_text("{not json")
    _use(monkeypatch, metric_mcid_path=str(p))
    r = ms.classify_change("updrs_total", 20, 25)
    assert r["threshold_basis"] == "provisional_relative_change"


def test_unknown_polarity_never_becomes_improving_or_deteriorating_under_mcid(monkeypatch, tmp_path):
    _use(monkeypatch, tmp_path, mcid={"mystery": {"mcid": 1.0, "approved_by": "x", "source": "y"}})
    assert ms.classify_change("mystery", 10, 20)["classification"] == "changed_direction_unknown"
