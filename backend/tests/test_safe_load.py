"""Restricted model loading (Phase 2, H): shipped artifacts load; code-execution pickles are refused."""
import os
import pickle
import joblib
import pytest
from pathlib import Path

from app.ml.safe_load import UnsafeModelError, restricted_joblib_load

MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "models"


@pytest.mark.parametrize("name", ["parkinsons_demo_classifier.joblib", "parkinsons_risk_classifier.joblib"])
def test_shipped_models_load_under_restriction(name):
    obj = restricted_joblib_load(MODEL_DIR / name)
    ref = joblib.load(MODEL_DIR / name)
    assert type(obj) is type(ref)


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > /dev/null",))


def test_malicious_pickle_is_refused(tmp_path):
    p = tmp_path / "evil.joblib"
    joblib.dump(_Evil(), p)
    with pytest.raises(UnsafeModelError):
        restricted_joblib_load(p)


def test_builtin_eval_is_refused(tmp_path):
    class E:
        def __reduce__(self):
            return (eval, ("1+1",))
    p = tmp_path / "e.joblib"
    joblib.dump(E(), p)
    with pytest.raises(UnsafeModelError):
        restricted_joblib_load(p)


def test_plain_numpy_roundtrip(tmp_path):
    import numpy as np
    p = tmp_path / "a.joblib"
    joblib.dump({"a": np.arange(5)}, p)
    assert restricted_joblib_load(p)["a"].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("module,name", [("os", "system"), ("subprocess", "Popen"), ("numpy", "load"),
                                         ("sklearn.datasets._base", "load_files"), ("builtins", "eval"),
                                         ("sklearn.utils._testing", "assert_all_finite"), ("posix", "system"),
                                         ("numpy.lib.npyio", "load")])
def test_policy_refuses_dangerous_names(module, name):
    from app.ml.safe_load import _check
    with pytest.raises(UnsafeModelError):
        _check(module, name)
