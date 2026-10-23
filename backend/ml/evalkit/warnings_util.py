"""Scoped warning handling for long numerical runs (Phase 6 audit finding F7).

Four modules used to call `warnings.filterwarnings("ignore")` at import time, which silences warnings for the WHOLE process, including
anything unrelated that imports them.  A convergence or deprecation warning inside a real validation run is information, so these helpers
(a) keep the console quiet only inside the decorated call, and (b) COUNT what was raised and put the counts in the result.
"""
from __future__ import annotations

import functools
import warnings
from collections import Counter
from contextlib import contextmanager


@contextmanager
def counting(sink: Counter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        sink[f"{w.category.__name__}: {str(w.message).splitlines()[0][:110]}"] += 1


def record_warnings(fn):
    """For functions returning a dict: adds `warnings_recorded` = {message: count}.  Nothing is suppressed globally."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        sink: Counter = Counter()
        with counting(sink):
            out = fn(*a, **k)
        if isinstance(out, dict):
            out = {**out, "warnings_recorded": dict(sink.most_common(25)), "n_warnings_recorded": int(sum(sink.values()))}
        return out
    return wrapper


def quiet(fn):
    """For command-line entry points that return an exit code: silences warnings inside the call only."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    return wrapper
