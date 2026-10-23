"""
Generate the results section of a validation report FROM machine-readable outputs, and verify it (Phase 5, Stage 7).

Every number that appears between the RESULTS markers is registered as a claim (file, JSON path, display string). `verify()` re-reads the
JSON, recomputes each display string, and rejects a report that (a) contains a number that is not a registered claim, (b) displays a
value different from the artifact, or (c) registers a claim from an experiment whose status is not 'completed'.
A NOT_RUN / FAILED / REFUSED experiment is rendered as its status and reason - with no numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
_NUM = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?(?![\w\-])")


class ReportMismatch(AssertionError):
    pass


def _get(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _fmt(v, fmt: str) -> str:
    if v is None:
        return "n/a"
    return str(int(v)) if fmt == "d" else f"{float(v):.{fmt}f}"


class _Builder:
    def __init__(self, results_dir: Path):
        self.dir, self.claims, self.lines, self._cache = Path(results_dir), [], [], {}

    def load(self, name: str):
        if name not in self._cache:
            p = self.dir / name
            self._cache[name] = json.loads(p.read_text()) if p.exists() else None
        return self._cache[name]

    def num(self, file: str, path: str, fmt: str = "3") -> str:
        v = _get(self.load(file), path)
        d = _fmt(v, fmt)
        self.claims.append({"file": file, "path": path, "fmt": fmt, "display": d})
        return d

    def add(self, line: str = ""):
        self.lines.append(line)


def _status_block(b: _Builder, kind: str, title: str) -> dict | None:
    done, not_run, failed = b.load(f"{kind}_p4a.json"), b.load(f"{kind}_NOT_RUN.json"), b.load(f"{kind}_FAILED.json")
    b.add(f"### {title}")
    if done and done.get("status") == "completed":
        b.add(f"Status: **completed**. Evidence label: `{done['evidence_status']}`. Protocol `{done['protocol_id']}` "
              f"(`{done['protocol_sha256'][:12]}…`). `clinically_validated` = {done['clinically_validated']}.")
        return done
    if failed:
        b.add(f"Status: **FAILED** - {failed.get('error', 'no reason recorded')}. No results are shown.")
    elif not_run:
        b.add(f"Status: **NOT RUN** - {not_run.get('reason', 'no reason recorded')} No results are shown.")
    else:
        b.add("Status: **NOT RUN** - no result file exists. No results are shown.")
    b.add("")
    return None


def _risk(b: _Builder):
    r = _status_block(b, "risk", "Risk classifier (concurrent worsening)")
    if not r:
        return
    f = "risk_p4a.json"
    seeds = [str(s) for s in r["seeds"]]
    first = seeds[0]
    b.add(f"Rows used: {b.num(f, 'dataset.n_rows', 'd')}; independent patients: {b.num(f, 'dataset.n_subjects', 'd')}; "
          f"positive rate: {b.num(f, 'dataset.positive_rate')}; seeds: {', '.join(b.num(f, f'seeds.{i}', 'd') for i in range(len(seeds)))} "
          "(they vary the fold assignment only; they are not independent datasets).")
    b.add("")
    b.add("| Feature set | AUC seed-mean | SD over seeds | min | max | AUC 95% patient-bootstrap CI (first seed) |")
    b.add("|---|---:|---:|---:|---:|---|")
    for name in r["dispersion_across_seeds"]:
        d, ci = f"dispersion_across_seeds.{name}", f"results_by_seed.{first}.{name}.pooled.roc_auc_cluster_bootstrap_ci95"
        b.add(f"| {name} | {b.num(f, d + '.mean')} | {b.num(f, d + '.sd')} | {b.num(f, d + '.min')} | {b.num(f, d + '.max')} | "
              f"{b.num(f, ci + '.lo')} to {b.num(f, ci + '.hi')} |")
    b.add("")
    b.add("| Voice added beyond reference | paired AUC difference (per seed) | 95% CI per seed | excludes 0 on every seed |")
    b.add("|---|---|---|---|")
    for ref, v in r["voice_incremental_value"]["by_reference"].items():
        base = f"voice_incremental_value.by_reference.{ref}"
        pts = ", ".join(b.num(f, f"{base}.point_by_seed.{i}") for i in range(len(seeds)))
        cis = "; ".join(f"{b.num(f, f'{base}.ci_lo_by_seed.{i}')} to {b.num(f, f'{base}.ci_hi_by_seed.{i}')}" for i in range(len(seeds)))
        b.add(f"| {ref} | {pts} | {cis} | {'yes' if v['meets_registered_rule'] else 'no'} |")
    b.add("")
    b.add(f"Registered rule met: **{'yes' if r['voice_incremental_value']['registered_rule_met'] else 'no'}**. "
          f"Reading as recorded by the runner: {r['voice_incremental_value']['reading']}")
    b.add("")


def _disease(b: _Builder):
    r = _status_block(b, "disease", "Disease classifier (PD vs healthy)")
    if not r:
        return
    f = "disease_p4a.json"
    first = str(r["seeds"][0])
    s = f"results_by_seed.{first}"
    b.add(f"Recordings: {b.num(f, 'dataset.n_rows', 'd')}; independent subjects: {b.num(f, 'dataset.n_subjects', 'd')} "
          f"(PD {b.num(f, 'dataset.subjects_pd', 'd')}, healthy {b.num(f, 'dataset.subjects_healthy', 'd')}).")
    b.add("")
    b.add("| Quantity (first seed) | Value |")
    b.add("|---|---:|")
    b.add(f"| Subject-level AUC | {b.num(f, s + '.subject_level.roc_auc')} |")
    b.add(f"| Subject-level AUC 95% subject-bootstrap CI | {b.num(f, s + '.subject_level.roc_auc_cluster_bootstrap_ci95.lo')} to "
          f"{b.num(f, s + '.subject_level.roc_auc_cluster_bootstrap_ci95.hi')} |")
    b.add(f"| Healthy subjects correct / total | {b.num(f, s + '.subject_level.healthy_subjects_correct', 'd')} / "
          f"{b.num(f, s + '.subject_level.healthy_subjects_total', 'd')} |")
    b.add(f"| Specificity exact 95% CI | {b.num(f, s + '.subject_level.specificity_clopper_pearson95.0')} to "
          f"{b.num(f, s + '.subject_level.specificity_clopper_pearson95.1')} |")
    b.add(f"| classic3_LR subject-level AUC | {b.num(f, s + '.baselines.classic3_LR.subject_level.roc_auc')} |")
    b.add(f"| dummy_prior AUC (constant score) | {b.num(f, s + '.baselines.dummy_prior.subject_level.roc_auc')} |")
    b.add(f"| Paired AUC difference vs classic3_LR | {b.num(f, s + '.paired_auc_difference_vs_classic3_LR_subject_level.point')} "
          f"({b.num(f, s + '.paired_auc_difference_vs_classic3_LR_subject_level.lo')} to {b.num(f, s + '.paired_auc_difference_vs_classic3_LR_subject_level.hi')}) |")
    b.add(f"| Permutation p (fixed pipeline) | {b.num(f, s + '.permutation_test_fixed_pipeline.p_value')} |")
    b.add("")


def render(results_dir: Path) -> tuple[str, list[dict]]:
    b = _Builder(results_dir)
    b.add(BEGIN)
    _risk(b)
    _disease(b)
    b.add(END)
    return "\n".join(b.lines) + "\n", b.claims


def verify(markdown: str, claims: list[dict], results_dir: Path) -> None:
    """Raises ReportMismatch unless every displayed number is a registered claim that equals the artifact."""
    b = _Builder(results_dir)
    for c in claims:
        v = _get(b.load(c["file"]), c["path"])
        if _fmt(v, c["fmt"]) != c["display"]:
            raise ReportMismatch(f"{c['file']}:{c['path']} displays {c['display']} but the artifact gives {_fmt(v, c['fmt'])}")
    if BEGIN not in markdown or END not in markdown:
        raise ReportMismatch("results markers missing")
    for f in {c["file"] for c in claims}:                         # numbers may only come from an experiment that actually completed
        if (b.load(f) or {}).get("status") != "completed":
            raise ReportMismatch(f"{f}: numbers are displayed for an experiment whose status is not 'completed'")
    region = markdown.split(BEGIN, 1)[1].split(END, 1)[0]
    shown = {c["display"] for c in claims}
    lines = region.splitlines()
    sep = re.compile(r"^\|[\-:| ]+\|$")
    for i, line in enumerate(lines):
        is_header = i + 1 < len(lines) and bool(sep.match(lines[i + 1]))
        if line.startswith(("Status:", "###", "Registered rule")) or not line.strip() or sep.match(line) or is_header:
            continue                                               # prose / table headers (labels such as '95%'); reasons may contain digits (HTTP 403)
        value_text = "|".join(line.split("|")[2:]) if line.startswith("|") else line      # the first column is a label, not a result
        for tok in _NUM.findall(value_text):
            if tok not in shown:
                raise ReportMismatch(f"number {tok!r} in the results region is not a registered claim: {line.strip()[:100]}")


def write_report(results_dir: Path, out_md: Path) -> list[dict]:
    md, claims = render(results_dir)
    verify(md, claims, results_dir)
    out_md.write_text("# Generated results (from machine-readable artifacts)\n\nEvery number below was read from the JSON files in this directory "
                      "and re-verified against them.\n\n" + md)
    (out_md.with_suffix(".claims.json")).write_text(json.dumps(claims, indent=1) + "\n")
    return claims
