# Phase 5: Reproducibility and Execution Instructions

One documented workflow. All commands run from `backend/`.

## 1. Environment

Reference environment used for Phase 5 (recorded in every `experiment_manifest.json`): Python 3.13.17, numpy 2.5.4,
pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, Linux x86_64.

The shipped artifacts were trained under scikit-learn 1.9.0 (inferred from an unpickle version warning; the artifacts do not
record it). Loading under 1.9.1 warns but works; this is a known provenance gap, not a verified compatibility guarantee.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock.txt     # exact pins (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1)
```

## 2. Tests

```
cd backend
python -m pytest -q -p no:cacheprovider                 # full backend suite
python -m pytest -q -p no:cacheprovider tests/phase5    # Phase 5 only
```

Historical baseline at `8e4f4d3`: 764 passed, 26 skipped. Phase 5 results are in `FINAL_READINESS_REPORT.md`. The 26 skips are live-service
tests (Supabase/Gemini/RLS). **No pytest test needs the real UCI files**, so a green pytest run is not real-data validation. Real-data
validation is the workflow in section 4, which exits 2 when the data are absent; CI must run it and treat exit code 2 as
"validation not run", never as success.

## 3. Obtaining the data (authorised path only)

The two public files are not downloadable from the sandbox (403). On a machine with access and rights to use them:

| File | Source |
|---|---|
| `parkinsons.data` | UCI Machine Learning Repository, dataset 174, https://archive.ics.uci.edu/dataset/174 |
| `parkinsons_updrs.data` | UCI Machine Learning Repository, dataset 189, https://archive.ics.uci.edu/dataset/189 |

Do not use unverified mirrors. Place the files in `backend/ml/data/cache/` or pass explicit paths (or set
`NEUROMAS_PARKINSONS_CSV` and `NEUROMAS_PARKINSONS_UPDRS_CSV`).

## 4. Run the registered validation

```
PYTHONPATH=. python -m ml.validation.dataset_manifest status
PYTHONPATH=. python -m ml.validation.workflow run \
    --disease-csv /path/to/parkinsons.data \
    --risk-csv /path/to/parkinsons_updrs.data \
    --out ml/validation/results/workflow
PYTHONPATH=. python -m ml.validation.workflow report --out ml/validation/results/workflow
```

What happens, in order, per dataset: schema and integrity checks (refuse on violation) -> manifest hash check (refuse on
mismatch) -> fingerprint authentication against the shipped artifact's scaler -> registered protocol `p4a-1`. An
unauthenticated file still runs but is labelled `UNAUTHENTICATED_USER_FILE_NOT_A_REPRODUCTION_OF_THE_REPORTED_FIGURES`. Add
`--fast` for a smoke test only (labelled `FAST_NOT_REGISTERED_PROTOCOL`; not evidence).

Exit codes: `0` completed, `1` failed, `2` not run (or incomplete), `3` refused. A mix of a completed and a not-run dataset is
`INCOMPLETE` and exits 2.

Outputs in `--out`: `experiment_manifest.json` (git commit and dirty flag, argv, environment, protocol id and hash, seeds,
status per experiment, output SHA-256), `disease_*.json` / `risk_*.json` results or `*_NOT_RUN.json`, data-quality and
exclusion reports, `GENERATED_RESULTS.md` and `GENERATED_RESULTS.claims.json`. Predictions and fold assignments go to
protected storage (`protected/`) because they are patient-level derivatives; do not commit them.

## 5. After a successful authenticated run

1. `python -m ml.validation.dataset_manifest pin ...` (requires authentication; records the file hashes).
2. Review `exclusion` and `data_quality` reports; record the real tie count, subject codes and `test_time` comparison.
3. If a protocol defect is found, write `p4a-2.json` (do not edit `p4a-1.json`) and label the analysis accordingly.
4. To obtain `COMPLETE` provenance, retrain with `PYTHONPATH=. python -m ml.training.train_final disease|risk --csv <authenticated file>`; it writes the sidecar.

## 6. Other commands

```
PYTHONPATH=. python -m ml.validation.phase4a artifacts --out <dir>     # no-data audit of shipped artifacts (deterministic)
PYTHONPATH=. python -m ml.validation.artifact_contract                  # regenerate legacy provenance sidecars
PYTHONPATH=. python -m ml.validation.power_planning --reps 20           # planning calculations (not evidence), ~7 min
```

The no-data artifact audit was re-run in Phase 5 and is identical to the committed Phase 4A result apart from the environment block.

## 7. Known limits of reproducibility

* Reported historical figures cannot be reproduced without the data.
* Seeds are fixed (base 20261023, five seeds); results from different seeds are split-sensitivity checks, not replications.
* Bootstrap and permutation counts: 2000 and 200 (registered).
* The workflow's `git_dirty` flag is true when it is run with uncommitted changes; a manifest from a dirty tree is not a clean
  provenance record.
