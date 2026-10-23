"""Audit finding F7: importing an ML validation module must not silence warnings process-wide, and warnings raised inside a run are counted."""
import subprocess
import sys
import warnings
from pathlib import Path

from ml.evalkit.warnings_util import counting, quiet, record_warnings

BACKEND = Path(__file__).resolve().parents[2]


def test_importing_the_validation_modules_installs_no_global_ignore_filter():
    code = (
        "import warnings\n"
        "before = set(map(repr, warnings.filters))\n"
        "import ml.validation.phase4a, ml.validation.phase4a_selftest, ml.validation.validate_models, ml.validation.phase6_report, ml.simulation.p3_statistical_validation\n"
        "new = [f for f in warnings.filters if repr(f) not in before]\n"
        "bad = [f for f in new if f[0] == 'ignore' and f[1] is None and f[2] is Warning and f[3] is None]\n"
        "print('BAD', bad)\nraise SystemExit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(BACKEND)})
    assert r.returncode == 0, r.stdout + r.stderr


def test_record_warnings_counts_what_was_raised_and_does_not_leak_a_filter():
    @record_warnings
    def f():
        warnings.warn("convergence did not happen", UserWarning)
        warnings.warn("convergence did not happen", UserWarning)
        warnings.warn("another", DeprecationWarning)
        return {"x": 1}

    n_before = len(warnings.filters)
    out = f()
    assert out["x"] == 1 and out["n_warnings_recorded"] == 3
    assert out["warnings_recorded"]["UserWarning: convergence did not happen"] == 2
    assert len(warnings.filters) == n_before


def test_quiet_silences_only_inside_the_call():
    @quiet
    def g():
        warnings.warn("hidden")
        return 7

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert g() == 7
        warnings.warn("visible")
    assert [str(x.message) for x in w] == ["visible"]


def test_registered_run_results_now_carry_the_warning_counts():
    from collections import Counter
    c = Counter()
    with counting(c):
        warnings.warn("x", RuntimeWarning)
    assert sum(c.values()) == 1
