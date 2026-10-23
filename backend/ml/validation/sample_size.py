"""
Sample-size arithmetic for the external validation protocol (Phase 6, Workstream 4/6).  PLANNING ARITHMETIC, NOT EVIDENCE.

Every function answers "how many INDEPENDENT UNITS (subjects) are needed for an interval of a stated width, if the true value were X".  X is
an assumption the caller must justify; none is estimated from data here.  Several recordings per subject do not add independent units;
`design_effect` shows how much they are worth under an assumed intra-subject correlation.

    PYTHONPATH=. python -m ml.validation.sample_size   # prints the illustrative tables used in docs/phase6/CLINICAL_VALIDATION_PROTOCOL.md
"""
from __future__ import annotations

import json
import math

from scipy.stats import beta

Z = 1.959964          # two-sided 95%


def n_for_proportion_wald(p: float, half_width: float, z: float = Z) -> int:
    """Subjects of the relevant class needed so that a Wald interval around an expected proportion p has the given half width."""
    _check_p(p)
    if not (0 < half_width < 1):
        raise ValueError("half_width must be in (0, 1)")
    return math.ceil(z * z * p * (1 - p) / (half_width ** 2))


def clopper_pearson_half_width(k: int, n: int, alpha: float = 0.05) -> float:
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float((hi - lo) / 2)


def n_for_proportion_exact(p: float, half_width: float, alpha: float = 0.05, n_max: int = 100_000) -> int:
    """Smallest n for which the exact (Clopper-Pearson) interval, at the expected count round(p*n), has at most the given half width.
    Not monotone in n at the margin (counts are integers), so the first n is returned AND every larger n up to n+50 is checked."""
    _check_p(p)
    n = 5
    while n <= n_max:
        if clopper_pearson_half_width(round(p * n), n, alpha) <= half_width and all(
                clopper_pearson_half_width(round(p * m), m, alpha) <= half_width for m in range(n, n + 51)):
            return n
        n += 1
    raise ValueError("n_max reached")


def subjects_for_sens_spec(prevalence: float, sensitivity: float, specificity: float, half_width: float) -> dict:
    """Total independent subjects so that BOTH sensitivity and specificity have a Wald half width <= half_width in a cohort with the given
    prevalence of the target condition.  The binding class decides."""
    _check_p(prevalence)
    n_pos, n_neg = n_for_proportion_wald(sensitivity, half_width), n_for_proportion_wald(specificity, half_width)
    need_from_pos, need_from_neg = _ceil(n_pos / prevalence), _ceil(n_neg / (1 - prevalence))
    return {"prevalence": prevalence, "assumed_sensitivity": sensitivity, "assumed_specificity": specificity, "half_width": half_width,
            "positives_needed": n_pos, "negatives_needed": n_neg, "total_subjects": max(need_from_pos, need_from_neg),
            "binding_class": "positive" if need_from_pos >= need_from_neg else "negative"}


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(var)


def subjects_for_auc_precision(auc: float, prevalence: float, half_width: float, n_max: int = 1_000_000) -> dict:
    """Smallest total of independent subjects (at the stated prevalence) such that the Hanley-McNeil 95% half width for the assumed AUC is
    <= half_width.  Normal approximation; reasonable above roughly 30 per class."""
    if not (0.5 < auc < 1):
        raise ValueError("auc must be in (0.5, 1)")
    _check_p(prevalence)
    n = 20
    while n <= n_max:
        n_pos, n_neg = max(2, round(n * prevalence)), max(2, n - round(n * prevalence))
        if Z * hanley_mcneil_se(auc, n_pos, n_neg) <= half_width:
            return {"assumed_auc": auc, "prevalence": prevalence, "half_width": half_width, "total_subjects": n, "positives": n_pos, "negatives": n_neg}
        n += 1
    raise ValueError("n_max reached")


def design_effect(recordings_per_subject: float, icc: float) -> float:
    """Kish design effect 1 + (m - 1) * rho: the factor by which clustering inflates the variance of a row-level proportion.  `icc` is an
    ASSUMPTION; it is not known for these data."""
    if recordings_per_subject < 1 or not (0 <= icc <= 1):
        raise ValueError("recordings_per_subject >= 1 and 0 <= icc <= 1 required")
    return 1 + (recordings_per_subject - 1) * icc


def effective_sample_size(n_rows: int, recordings_per_subject: float, icc: float) -> float:
    return n_rows / design_effect(recordings_per_subject, icc)


def _ceil(x: float) -> int:
    """ceil that ignores binary floating-point noise (246 / (1 - 0.9) is 2460.0000000000005, not 2461 subjects)."""
    return math.ceil(round(x, 9))


def _check_p(p: float) -> None:
    if not (0 < p < 1):
        raise ValueError("a proportion must be in (0, 1)")


def illustrative_tables() -> dict:
    """The tables quoted in the validation protocol.  ASSUMED values only."""
    return {
        "label": "PLANNING_ARITHMETIC_NOT_EVIDENCE",
        "sens_spec": [subjects_for_sens_spec(prev, 0.85, 0.80, hw) for prev in (0.10, 0.25, 0.50) for hw in (0.10, 0.05)],
        "auc_precision": [subjects_for_auc_precision(a, prev, 0.05) for a in (0.70, 0.80, 0.90) for prev in (0.25, 0.50)],
        "design_effect_examples": [{"recordings_per_subject": m, "icc": r, "design_effect": design_effect(m, r), "rows_worth_of_subjects_per_100_rows": 100 / design_effect(m, r)}
                                   for m in (6, 139) for r in (0.1, 0.5, 0.9)],
    }


if __name__ == "__main__":
    print(json.dumps(illustrative_tables(), indent=1))
