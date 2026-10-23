"""
Defence-in-depth for loading scikit-learn artifacts (Phase 2, H).

joblib/pickle files can execute arbitrary code when loaded. The first line of defence is the SHA-256 manifest check
(ml/models/MANIFEST.json, enforced before any load). This module adds a second: a RESTRICTED unpickler that only
resolves classes from numpy / scipy / scikit-learn / joblib / a few builtins, and refuses names that smell like code
execution or I/O. A tampered file whose hash somehow matched (or a manifest edited together with the artifact) still
cannot call os.system, subprocess, exec, eval, numpy.load, etc.

It is an allow-list by module PREFIX plus a deny-list of dangerous tokens; it is not a proof of safety (a gadget inside
an allowed library is conceivable). The durable fix is a non-pickle format (ONNX / skops with trusted types), which
needs the training data to re-export and re-validate; see docs/phase2/SECURITY_AND_PRIVACY.md.

Fails CLOSED: if joblib's internal reader cannot be used, loading raises instead of falling back to joblib.load.
"""
from __future__ import annotations

from pathlib import Path

ALLOWED_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "slice", "complex", "bytearray", "object", "int",
                    "float", "str", "bool", "bytes", "range"}
ALLOWED_EXACT = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("copyreg", "_reconstructor"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__generator_ctor"),
    ("numpy.random._pickle", "__bit_generator_ctor"),
    # Phase 5 (E06): a fitted HistGradientBoostingClassifier stores a numpy Generator, whose state pickles as these three names.
    # They are inert numeric-state classes; added as EXACT names (no prefix widening). Without them a retrained model whose winner is
    # the boosting candidate could not be loaded at all.
    ("numpy.random._pcg64", "PCG64"), ("numpy.random.bit_generator", "SeedSequence"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
}
# scikit-learn / scipy.sparse are allowed by prefix, minus modules that perform I/O or are test helpers.
ALLOWED_PREFIXES = ("sklearn.", "scipy.sparse")
DENY_MODULE_TOKENS = ("test", "datasets", "externals", "_loadarff", "io", "ctypes", "subprocess", "os", "sys", "shutil",
                      "socket", "pickle", "marshal", "builtins")


class UnsafeModelError(Exception):
    pass


def _check(module: str, name: str) -> None:
    if module == "builtins":
        if name in ALLOWED_BUILTINS:
            return
        raise UnsafeModelError(f"builtins.{name} is not allowed")
    if (module, name) in ALLOWED_EXACT:
        return
    if module.startswith(ALLOWED_PREFIXES):
        parts = module.lower().split(".")
        if not any(tok == part or (tok in ("test", "datasets") and tok in part) for part in parts
                   for tok in DENY_MODULE_TOKENS):
            return
    raise UnsafeModelError(f"refusing to resolve {module}.{name} while loading a model artifact")


def restricted_joblib_load(path: str | Path):
    from joblib import numpy_pickle as npk

    for attr in ("NumpyUnpickler", "_validate_fileobject_and_memmap"):
        if not hasattr(npk, attr):
            raise UnsafeModelError("joblib internals changed; refusing to fall back to an unrestricted load")

    class _Restricted(npk.NumpyUnpickler):
        def find_class(self, module, name):
            _check(module, name)
            return super().find_class(module, name)

    p = Path(path)
    with open(p, "rb") as f:
        with npk._validate_fileobject_and_memmap(f, str(p), None) as (fobj, _):
            if isinstance(fobj, str):
                raise UnsafeModelError("legacy (pre-0.10) joblib format is not supported")
            return _Restricted(str(p), fobj, True, mmap_mode=None).load()
