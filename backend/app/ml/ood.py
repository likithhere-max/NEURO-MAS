"""
Input-domain validation and out-of-distribution (OOD) abstention for ML inference (Phase 2, C4).

A prediction is only meaningful for inputs that look like the data the model was trained on. This module
decides, BEFORE calling `predict`, whether the input is inside the validated domain. If not, the API returns a
structured ABSTENTION instead of a number.

Layers (all deterministic; the result lists exactly which ones ran):

  L1 contract      feature names/order exactly as trained, numeric, finite                -> abstain_invalid_input
  L2 physical      hard constraints from the definitions of the measures (e.g. jitter >= 0,
                   Flo <= Fo <= Fhi, 0 <= UPDRS <= 176)                                   -> abstain_invalid_input
  L3 identities    exact algebraic relations between measures (DDP = 3*RAP, DDA = 3*APQ3);
                   a violation means the vector was not produced by one real analysis      -> abstain_invalid_input
  L4 marginal      each feature must lie inside [train_min, train_max] widened by a margin
                   (fitted domain) or within +/- z_limit SD of the training mean taken from
                   the model's own StandardScaler (provisional domain)                     -> abstain_out_of_domain
  L5 joint         optional Mahalanobis distance (Ledoit-Wolf shrinkage covariance) against a
                   threshold calibrated by leave-one-group-out on the training data         -> abstain_out_of_domain

What this does NOT do: it does not say the prediction is right, only that the input is not obviously foreign.
L4/L5 with Gaussian-style distances are NOT valid density estimates for heavy-tailed acoustic features; the
joint test is a diagnostic whose false-rejection / missed-OOD rates must be measured
(`evaluate_domain_detector`) before it is trusted.

Two domain sources:
  * "fitted"                    from `fit_domain(X_train, groups)`; JSON-serialisable; produced by the training
                                script on real data (NOT RUN in the sandbox: dataset unreachable).
  * "scaler_moments_provisional" derived from the StandardScaler inside the shipped model (training mean/SD of
                                each feature). Weaker: no min/max, no covariance.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

DOMAIN_VERSION = "ood-1.0"


# ----------------------------------------------------------------------------- result types
@dataclass(frozen=True)
class Reason:
    code: str            # e.g. "missing_feature", "non_finite", "physical_bound", "identity", "range", "mahalanobis"
    feature: str | None
    detail: str

    def as_dict(self) -> dict:
        return {"code": self.code, "feature": self.feature, "detail": self.detail}


@dataclass
class DomainResult:
    status: str                              # in_domain | abstain_invalid_input | abstain_out_of_domain
    reasons: list[Reason] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    max_abs_z: float | None = None
    mahalanobis_d2: float | None = None
    domain_source: str = ""
    domain_version: str = DOMAIN_VERSION

    @property
    def in_domain(self) -> bool:
        return self.status == "in_domain"

    def as_dict(self) -> dict:
        return {"status": self.status, "in_domain": self.in_domain, "reasons": [r.as_dict() for r in self.reasons],
                "checks_run": self.checks_run, "max_abs_z": self.max_abs_z, "mahalanobis_d2": self.mahalanobis_d2,
                "domain_source": self.domain_source, "domain_version": self.domain_version}


# ----------------------------------------------------------------------------- constraints
@dataclass(frozen=True)
class Bound:
    feature: str
    lo: float | None = None
    hi: float | None = None
    why: str = ""


@dataclass(frozen=True)
class Identity:
    """a == factor * b within abs_tol + rel_tol*|expected|."""
    a: str
    b: str
    factor: float
    abs_tol: float
    rel_tol: float
    why: str = ""


@dataclass(frozen=True)
class Ordering:
    """features[0] <= features[1] <= ... (with tolerance)."""
    features: tuple[str, ...]
    why: str = ""
    tol: float = 1e-9


def _voice_constraints():
    nonneg = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
              "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "PPE"]
    bounds = [Bound(f, lo=0.0, why="perturbation/noise measures are non-negative by definition") for f in nonneg]
    # Plausibility (physiological) bounds for fundamental frequency of human phonation: engineering bounds
    bounds += [Bound(f, lo=30.0, hi=1000.0, why="plausible human F0 range (engineering bound, generous)")
               for f in ("MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)")]
    bounds += [Bound("RPDE", lo=0.0, hi=1.0, why="normalised entropy"), Bound("D2", lo=0.0, why="correlation dimension")]
    ids = [Identity("Jitter:DDP", "MDVP:RAP", 3.0, 1e-4, 0.05, "Praat defines DDP = 3 x RAP"),
           Identity("Shimmer:DDA", "Shimmer:APQ3", 3.0, 1e-4, 0.05, "Praat defines DDA = 3 x APQ3")]
    orders = [Ordering(("MDVP:Flo(Hz)", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"), "min <= mean <= max of F0", tol=1e-6)]
    return bounds, ids, orders


def _risk_constraints():
    bounds = [Bound("test_time", lo=0.0, why="days since baseline visit cannot be negative"),
              Bound("baseline_total_UPDRS", lo=0.0, hi=176.0, why="total UPDRS scale range")]
    # Deltas of linear identities remain linear identities
    ids = [Identity("delta_Jitter:DDP", "delta_Jitter:RAP", 3.0, 1e-4, 0.05, "DDP = 3 x RAP => same for differences"),
           Identity("delta_Shimmer:DDA", "delta_Shimmer:APQ3", 3.0, 1e-4, 0.05, "DDA = 3 x APQ3 => same for differences")]
    return bounds, ids, []


CONSTRAINTS: dict[str, Callable] = {"voice": _voice_constraints, "risk": _risk_constraints,
                                    "generic": lambda: ([], [], [])}


# ----------------------------------------------------------------------------- the domain object
@dataclass
class InputDomain:
    kind: str                                    # "voice" | "risk"
    feature_order: list[str]
    source: str                                  # "fitted" | "scaler_moments_provisional"
    mean: dict[str, float]
    sd: dict[str, float]
    train_min: dict[str, float] | None = None
    train_max: dict[str, float] | None = None
    range_margin: float = 0.10                   # widen [min,max] by this fraction of the range
    z_limit: float = 8.0                         # used when train_min/max unavailable
    center: list[float] | None = None            # Mahalanobis (standardised space)
    precision: list[list[float]] | None = None
    d2_threshold: float | None = None
    d2_calibration: dict | None = None
    n_train_rows: int | None = None

    # -- construction
    @classmethod
    def from_scaler(cls, kind: str, feature_order: Sequence[str], scaler, *, z_limit: float = 8.0) -> "InputDomain":
        mean, scale = getattr(scaler, "mean_", None), getattr(scaler, "scale_", None)
        if mean is None or scale is None or len(mean) != len(feature_order):
            raise ValueError("scaler does not match the feature order; cannot derive a provisional domain")
        sd = np.where(np.asarray(scale) > 0, scale, np.nan)
        return cls(kind=kind, feature_order=list(feature_order), source="scaler_moments_provisional",
                   mean=dict(zip(feature_order, map(float, mean))), sd=dict(zip(feature_order, map(float, sd))), z_limit=z_limit)

    def to_json(self) -> str:
        def clean(o):
            if isinstance(o, float) and not math.isfinite(o):
                return None
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [clean(v) for v in o]
            return o
        return json.dumps(clean(self.__dict__), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "InputDomain":
        return cls(**json.loads(text))

    # -- the check
    def check(self, row: dict) -> DomainResult:
        res = DomainResult(status="in_domain", domain_source=self.source)
        bounds, ids, orders = CONSTRAINTS[self.kind]()

        def bad(status, code, feature, detail):
            res.reasons.append(Reason(code, feature, detail))
            if res.status == "in_domain" or status == "abstain_invalid_input":
                res.status = status  # invalid input outranks out-of-domain

        # L1 contract
        res.checks_run.append("L1_contract")
        if not isinstance(row, dict):
            bad("abstain_invalid_input", "not_a_mapping", None, "input must be a mapping of feature name to value")
            return res
        missing = [f for f in self.feature_order if f not in row]
        extra = [f for f in row if f not in self.feature_order]
        for f in missing:
            bad("abstain_invalid_input", "missing_feature", f, "required feature absent")
        for f in extra:
            bad("abstain_invalid_input", "unexpected_feature", f, "feature not part of the trained contract")
        vals: dict[str, float] = {}
        for f in self.feature_order:
            if f not in row:
                continue
            v = row[f]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                bad("abstain_invalid_input", "not_numeric", f, f"got {type(v).__name__}")
            elif not math.isfinite(float(v)):
                bad("abstain_invalid_input", "non_finite", f, "NaN/inf")
            else:
                vals[f] = float(v)
        if res.status != "in_domain":
            return res  # later layers need a complete numeric vector

        # L2 physical
        res.checks_run.append("L2_physical")
        for b in bounds:
            if b.feature in vals:
                v = vals[b.feature]
                if (b.lo is not None and v < b.lo) or (b.hi is not None and v > b.hi):
                    bad("abstain_invalid_input", "physical_bound", b.feature, f"{v} outside [{b.lo}, {b.hi}] ({b.why})")
        for o in orders:
            seq = [vals.get(f) for f in o.features]
            if all(s is not None for s in seq) and any(seq[i] > seq[i + 1] + o.tol for i in range(len(seq) - 1)):
                bad("abstain_invalid_input", "ordering", None, f"{' <= '.join(o.features)} violated ({o.why})")

        # L3 identities
        res.checks_run.append("L3_identities")
        for idn in ids:
            if idn.a in vals and idn.b in vals:
                expected = idn.factor * vals[idn.b]
                if abs(vals[idn.a] - expected) > idn.abs_tol + idn.rel_tol * abs(expected):
                    bad("abstain_invalid_input", "identity", idn.a,
                        f"{idn.a}={vals[idn.a]:.6g} but {idn.factor:g} x {idn.b}={expected:.6g} ({idn.why})")
        if res.status == "abstain_invalid_input":
            return res

        # L4 marginal
        res.checks_run.append("L4_marginal_range" if self.train_min else "L4_marginal_z_provisional")
        zs = []
        for f in self.feature_order:
            v = vals[f]
            sd = self.sd.get(f)
            if sd and math.isfinite(sd) and sd > 0:
                zs.append(abs((v - self.mean[f]) / sd))
            if self.train_min is not None:
                lo, hi = self.train_min[f], self.train_max[f]
                pad = self.range_margin * (hi - lo)
                if v < lo - pad or v > hi + pad:
                    bad("abstain_out_of_domain", "range", f, f"{v:.6g} outside training range [{lo:.6g}, {hi:.6g}] +/- {self.range_margin:.0%}")
            else:
                sdv = self.sd.get(f)
                if sdv and math.isfinite(sdv) and sdv > 0 and abs((v - self.mean[f]) / sdv) > self.z_limit:
                    bad("abstain_out_of_domain", "range", f, f"|z|={abs((v - self.mean[f]) / sdv):.1f} > {self.z_limit} vs training mean/SD")
        res.max_abs_z = float(max(zs)) if zs else None

        # L5 joint
        if self.precision is not None and self.d2_threshold is not None:
            res.checks_run.append("L5_mahalanobis_ledoit_wolf")
            z = np.array([((vals[f] - self.mean[f]) / self.sd[f]) if (self.sd.get(f) and math.isfinite(self.sd[f]) and self.sd[f] > 0) else 0.0
                          for f in self.feature_order])
            d = z - np.asarray(self.center)
            d2 = float(d @ np.asarray(self.precision) @ d)
            res.mahalanobis_d2 = d2
            if d2 > self.d2_threshold:
                bad("abstain_out_of_domain", "mahalanobis", None,
                    f"d^2={d2:.2f} > calibrated threshold {self.d2_threshold:.2f}")
        return res


# ----------------------------------------------------------------------------- fitting + evaluation
def fit_domain(kind: str, X, groups=None, *, range_margin: float = 0.10, quantile: float = 0.99, z_limit: float = 8.0) -> InputDomain:
    """Fit a domain from TRAINING data only. If `groups` is given the d^2 threshold is calibrated by
    leave-one-group-out (covariance fitted without the held-out group), so it reflects how far an UNSEEN
    patient typically sits, not how close the training rows are to themselves."""
    import pandas as pd
    from sklearn.covariance import LedoitWolf

    X = pd.DataFrame(X)
    cols = list(X.columns)
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("training data contain non-finite values")
    mean, sd = X.mean(), X.std(ddof=1).replace(0, np.nan)
    Z = ((X - mean) / sd).fillna(0.0).to_numpy()

    def d2_of(model, A):
        d = A - model.location_
        return np.einsum("ij,jk,ik->i", d, model.precision_, d)

    full = LedoitWolf().fit(Z)
    cal = {"method": "ledoit_wolf_mahalanobis", "quantile": quantile, "n_rows": int(len(Z)), "n_features": len(cols)}
    if groups is not None:
        g = np.asarray(groups)
        held = np.zeros(len(Z))
        for grp in dict.fromkeys(g.tolist()):
            m = g == grp
            if (~m).sum() <= Z.shape[1] // 2:
                continue
            held[m] = d2_of(LedoitWolf().fit(Z[~m]), Z[m])
        thr = float(np.quantile(held[held > 0], quantile)) if (held > 0).any() else float(np.quantile(d2_of(full, Z), quantile))
        cal["calibration"] = "leave-one-group-out"
    else:
        thr = float(np.quantile(d2_of(full, Z), quantile))
        cal["calibration"] = "in-sample (optimistic: group labels were not supplied)"
    cal["threshold"] = thr
    return InputDomain(kind=kind, feature_order=cols, source="fitted", mean={c: float(mean[c]) for c in cols},
                       sd={c: float(sd[c]) if not math.isnan(sd[c]) else float("nan") for c in cols},
                       train_min={c: float(X[c].min()) for c in cols}, train_max={c: float(X[c].max()) for c in cols},
                       range_margin=range_margin, z_limit=z_limit, center=full.location_.tolist(),
                       precision=full.precision_.tolist(), d2_threshold=thr, d2_calibration=cal, n_train_rows=int(len(X)))


def evaluate_domain_detector(domain: InputDomain, in_dist: Iterable[dict], perturbations: dict[str, Callable[[dict], dict]]) -> dict:
    """Controlled evaluation: false-rejection rate on held-out in-distribution rows and missed-OOD rate for each
    named perturbation. `in_dist` MUST be rows the domain was not fitted on; perturbations are applied to them."""
    rows = list(in_dist)
    base = [domain.check(r) for r in rows]
    fr = sum(not b.in_domain for b in base)
    out = {"n_in_distribution": len(rows), "false_rejection_rate": fr / len(rows) if rows else None,
           "false_rejections": fr, "perturbations": {}}
    for name, fn in perturbations.items():
        results = [domain.check(fn(dict(r))) for r in rows]
        detected = sum(not x.in_domain for x in results)
        out["perturbations"][name] = {"n": len(rows), "detected": detected,
                                      "missed_ood_rate": 1 - detected / len(rows) if rows else None}
    return out
