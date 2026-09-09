"use client";

// Longitudinal / personal baseline tab -- frontend surface for
// longitudinal_agent's two real, already-computed signals (see
// backend/app/agents/longitudinal.py and services/repository.py):
//   1. baseline-vs-current percentage-change per tracked metric
//   2. personalized within-patient anomaly detection (z-score against THIS
//      patient's own metric_readings history -- never a population model)
// Both existed on the backend with zero frontend before this tab; nothing
// here recomputes anything -- it only logs raw inputs and displays the
// agent's own findings/reasoning_summary.

import { useState } from "react";
import Link from "next/link";
import { usePatient } from "@/components/PatientContext";
import { api } from "@/lib/api";

const COMMON_METRICS = [
  "tremor_amplitude",
  "gait_speed_m_s",
  "sleep_hours",
  "medication_adherence_pct",
  "seizure_frequency_per_week",
  "balance_score",
];

const SEVERITY_BADGE: Record<string, string> = {
  critical: "badge-critical",
  high: "badge-high",
  warn: "badge-warn",
  info: "badge-info",
};

function cleanLabel(label: string) {
  // backend finding labels look like "trend:tremor_amplitude" or
  // "personal_anomaly:gait_speed_m_s" -- split the prefix out for display.
  const [kind, metric] = label.split(":");
  if (kind === "personal_anomaly") return `${metric} — personal anomaly`;
  if (kind === "trend") return `${metric} — trend`;
  return label;
}

export default function LongitudinalPage() {
  const { patientId, latestWorkflow, refreshAll } = usePatient();

  const [metric, setMetric] = useState(COMMON_METRICS[0]);
  const [customMetric, setCustomMetric] = useState("");
  const [value, setValue] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justLogged, setJustLogged] = useState<string | null>(null);

  const effectiveMetric = metric === "__custom__" ? customMetric.trim() : metric;

  async function logReading(e: React.FormEvent) {
    e.preventDefault();
    if (!effectiveMetric || !value) {
      setError("Pick a metric and enter a value.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await api.addMetricReading(patientId, { metric: effectiveMetric, value: Number(value) });
      setJustLogged(effectiveMetric);
      setValue("");
      await refreshAll();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  // -- Baseline vs. current (the OTHER, instant signal -- see the card below) --
  const [baselineMetric, setBaselineMetric] = useState(COMMON_METRICS[0]);
  const [baselineCustomMetric, setBaselineCustomMetric] = useState("");
  const [baselineValue, setBaselineValue] = useState("");
  const [currentValue, setCurrentValue] = useState("");
  const [baselineSubmitting, setBaselineSubmitting] = useState(false);
  const [baselineError, setBaselineError] = useState<string | null>(null);
  const [baselineJustSet, setBaselineJustSet] = useState<string | null>(null);

  const effectiveBaselineMetric = baselineMetric === "__custom__" ? baselineCustomMetric.trim() : baselineMetric;

  async function setBaseline(e: React.FormEvent) {
    e.preventDefault();
    if (!effectiveBaselineMetric || !baselineValue || !currentValue) {
      setBaselineError("Pick a metric and enter both a baseline and a current value.");
      return;
    }
    setBaselineSubmitting(true);
    setBaselineError(null);
    try {
      await api.addLongitudinalMetric(patientId, {
        metric: effectiveBaselineMetric,
        baseline_value: Number(baselineValue),
        current_value: Number(currentValue),
        window: "since baseline",
      });
      setBaselineJustSet(effectiveBaselineMetric);
      await refreshAll();
    } catch (err: any) {
      setBaselineError(err.message);
    } finally {
      setBaselineSubmitting(false);
    }
  }

  const agentResult = latestWorkflow?.agent_results?.longitudinal_agent;
  const findings = agentResult?.findings || [];

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold mb-1">Longitudinal &amp; personal baseline</h1>
        <p className="text-sm text-neuro-muted max-w-2xl">
          Track a metric over time. Every comparison here is against this patient&apos;s own history — never a
          population average — computed by real z-score statistics, not an LLM guess.
        </p>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Log a reading</h2>
        <form onSubmit={logReading} className="flex flex-wrap items-end gap-3">
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Metric</div>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            >
              {COMMON_METRICS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value="__custom__">Custom…</option>
            </select>
          </label>
          {metric === "__custom__" && (
            <label className="text-xs">
              <div className="text-neuro-muted mb-1">Custom metric name</div>
              <input
                value={customMetric}
                onChange={(e) => setCustomMetric(e.target.value)}
                placeholder="e.g. handwriting_size_mm"
                className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
              />
            </label>
          )}
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Value</div>
            <input
              type="number"
              step="any"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm w-32"
            />
          </label>
          <button
            type="submit"
            disabled={submitting}
            className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50"
          >
            {submitting ? "Logging…" : "Log reading"}
          </button>
        </form>
        {error && <div className="text-neuro-critical text-sm mt-2">{error}</div>}
        {justLogged && !error && (
          <div className="text-xs text-neuro-ok mt-3">
            Logged. This feeds personal-anomaly detection, which needs at least 3 prior readings for the same
            metric before it can flag anything unusual — for an instant result instead, use "Set baseline &amp;
            current value" below.
          </div>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Set baseline &amp; current value</h2>
        <p className="text-xs text-neuro-muted mb-3">
          The faster signal — no history needed. Set where this metric started and where it is now; the agent
          classifies it as stable, improving, or deteriorating immediately on the next assessment run.
        </p>
        <form onSubmit={setBaseline} className="flex flex-wrap items-end gap-3">
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Metric</div>
            <select
              value={baselineMetric}
              onChange={(e) => setBaselineMetric(e.target.value)}
              className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            >
              {COMMON_METRICS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value="__custom__">Custom…</option>
            </select>
          </label>
          {baselineMetric === "__custom__" && (
            <label className="text-xs">
              <div className="text-neuro-muted mb-1">Custom metric name</div>
              <input
                value={baselineCustomMetric}
                onChange={(e) => setBaselineCustomMetric(e.target.value)}
                placeholder="e.g. handwriting_size_mm"
                className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
              />
            </label>
          )}
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Baseline value</div>
            <input
              type="number"
              step="any"
              value={baselineValue}
              onChange={(e) => setBaselineValue(e.target.value)}
              className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm w-28"
            />
          </label>
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Current value</div>
            <input
              type="number"
              step="any"
              value={currentValue}
              onChange={(e) => setCurrentValue(e.target.value)}
              className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm w-28"
            />
          </label>
          <button
            type="submit"
            disabled={baselineSubmitting}
            className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50"
          >
            {baselineSubmitting ? "Saving…" : "Set values"}
          </button>
        </form>
        {baselineError && <div className="text-neuro-critical text-sm mt-2">{baselineError}</div>}
        {baselineJustSet && !baselineError && (
          <div className="text-xs text-neuro-ok mt-3">
            Saved. Go to{" "}
            <Link href={`/patient/${patientId}/assessment`} className="text-neuro-accent">
              Symptom Assessment
            </Link>{" "}
            and click Update Assessment — the trend analysis below will show this metric&apos;s classification right
            away.
          </div>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">Trend analysis</h2>
        {!latestWorkflow && (
          <div className="text-sm text-neuro-muted">No assessment has run yet, so there&apos;s nothing to compare.</div>
        )}
        {latestWorkflow && !agentResult && (
          <div className="text-sm text-neuro-muted">
            This workflow run didn&apos;t include longitudinal data yet.
          </div>
        )}
        {agentResult && (
          <div className="space-y-3">
            <p className="text-sm text-neuro-text">{agentResult.reasoning_summary}</p>
            {findings.length === 0 && (
              <div className="text-xs text-neuro-muted">
                Nothing to show yet — set a baseline &amp; current value above for an instant classification, or log
                3+ readings for the same metric to enable personal-anomaly detection, then re-run the assessment.
              </div>
            )}
            <div className="space-y-2">
              {findings.map((f: any, i: number) => (
                <div key={i} className="flex items-center justify-between border border-neuro-border rounded-lg px-3 py-2">
                  <div>
                    <div className="text-sm font-medium">{cleanLabel(f.label)}</div>
                    <div className="text-xs text-neuro-muted">{f.detail}</div>
                  </div>
                  <span className={`badge ${SEVERITY_BADGE[f.severity] || "badge-info"}`}>{f.severity}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
