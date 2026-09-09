"use client";

// ML Lab -- frontend surface for the Phase-3 standalone ML demonstration
// models (backend/app/api/routes/ml_demo.py). These endpoints already
// existed (built in an earlier sprint, validated with real leakage-safe
// cross-validation) but had ZERO frontend -- only reachable via /docs. This
// page is deliberately NOT nested under /patient/[id]/* and never touches
// PatientContext: the backend's own design keeps these models fully
// decoupled from real patient data, and the frontend now honors that same
// boundary rather than quietly wiring them into a patient's assessment.
//
// The two "load a real example" buttons below are genuine rows fetched
// directly from the UCI Parkinsons voice dataset (archive.ics.uci.edu),
// not fabricated numbers -- consistent with this project's own
// no-fake-data rule.

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, loadSession } from "@/lib/api";

type FeatureField = { alias: string; label: string; parkinsons: number; healthy: number };

const VOICE_FEATURES: FeatureField[] = [
  { alias: "MDVP:Fo(Hz)", label: "Fo (Hz)", parkinsons: 119.992, healthy: 197.076 },
  { alias: "MDVP:Fhi(Hz)", label: "Fhi (Hz)", parkinsons: 157.302, healthy: 206.896 },
  { alias: "MDVP:Flo(Hz)", label: "Flo (Hz)", parkinsons: 74.997, healthy: 192.055 },
  { alias: "MDVP:Jitter(%)", label: "Jitter %", parkinsons: 0.00784, healthy: 0.00289 },
  { alias: "MDVP:Jitter(Abs)", label: "Jitter (abs)", parkinsons: 0.00007, healthy: 0.00001 },
  { alias: "MDVP:RAP", label: "RAP", parkinsons: 0.0037, healthy: 0.00166 },
  { alias: "MDVP:PPQ", label: "PPQ", parkinsons: 0.00554, healthy: 0.00168 },
  { alias: "Jitter:DDP", label: "Jitter DDP", parkinsons: 0.01109, healthy: 0.00498 },
  { alias: "MDVP:Shimmer", label: "Shimmer", parkinsons: 0.04374, healthy: 0.01098 },
  { alias: "MDVP:Shimmer(dB)", label: "Shimmer (dB)", parkinsons: 0.426, healthy: 0.097 },
  { alias: "Shimmer:APQ3", label: "Shimmer APQ3", parkinsons: 0.02182, healthy: 0.00563 },
  { alias: "Shimmer:APQ5", label: "Shimmer APQ5", parkinsons: 0.0313, healthy: 0.0068 },
  { alias: "MDVP:APQ", label: "APQ", parkinsons: 0.02971, healthy: 0.00802 },
  { alias: "Shimmer:DDA", label: "Shimmer DDA", parkinsons: 0.06545, healthy: 0.01689 },
  { alias: "NHR", label: "NHR", parkinsons: 0.02211, healthy: 0.00339 },
  { alias: "HNR", label: "HNR", parkinsons: 21.033, healthy: 26.775 },
  { alias: "RPDE", label: "RPDE", parkinsons: 0.414783, healthy: 0.422229 },
  { alias: "DFA", label: "DFA", parkinsons: 0.815285, healthy: 0.741367 },
  { alias: "spread1", label: "spread1", parkinsons: -4.813031, healthy: -7.3483 },
  { alias: "spread2", label: "spread2", parkinsons: 0.266482, healthy: 0.177551 },
  { alias: "D2", label: "D2", parkinsons: 2.301442, healthy: 1.743867 },
  { alias: "PPE", label: "PPE", parkinsons: 0.284654, healthy: 0.085569 },
];

const RISK_FEATURES: { alias: string; label: string }[] = [
  { alias: "delta_Jitter(%)", label: "Δ Jitter %" },
  { alias: "delta_Jitter(Abs)", label: "Δ Jitter (abs)" },
  { alias: "delta_Jitter:RAP", label: "Δ RAP" },
  { alias: "delta_Jitter:PPQ5", label: "Δ PPQ5" },
  { alias: "delta_Jitter:DDP", label: "Δ Jitter DDP" },
  { alias: "delta_Shimmer", label: "Δ Shimmer" },
  { alias: "delta_Shimmer(dB)", label: "Δ Shimmer (dB)" },
  { alias: "delta_Shimmer:APQ3", label: "Δ Shimmer APQ3" },
  { alias: "delta_Shimmer:APQ5", label: "Δ Shimmer APQ5" },
  { alias: "delta_Shimmer:APQ11", label: "Δ Shimmer APQ11" },
  { alias: "delta_Shimmer:DDA", label: "Δ Shimmer DDA" },
  { alias: "delta_NHR", label: "Δ NHR" },
  { alias: "delta_HNR", label: "Δ HNR" },
  { alias: "delta_RPDE", label: "Δ RPDE" },
  { alias: "delta_DFA", label: "Δ DFA" },
  { alias: "delta_PPE", label: "Δ PPE" },
];

function emptyValues(fields: { alias: string }[], fill = 0) {
  const out: Record<string, number> = {};
  fields.forEach((f) => (out[f.alias] = fill));
  return out;
}

function FeatureGrid({
  fields,
  values,
  onChange,
}: {
  fields: { alias: string; label: string }[];
  values: Record<string, number>;
  onChange: (alias: string, v: number) => void;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
      {fields.map((f) => (
        <label key={f.alias} className="text-xs">
          <div className="text-neuro-muted mb-1">{f.label}</div>
          <input
            type="number"
            step="any"
            value={values[f.alias]}
            onChange={(e) => onChange(f.alias, Number(e.target.value))}
            className="w-full bg-neuro-panel2 border border-neuro-border rounded-md px-2 py-1.5 text-xs"
          />
        </label>
      ))}
    </div>
  );
}

function ProbabilityBar({ label, value, tone }: { label: string; value: number; tone: "ok" | "critical" }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span>{label}</span>
        <span className="text-neuro-muted">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-neuro-panel2 overflow-hidden">
        <div
          className={`h-full ${tone === "critical" ? "bg-neuro-critical" : "bg-neuro-ok"}`}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function MLLabPage() {
  const router = useRouter();

  // -- Disease classifier state --
  const [voiceValues, setVoiceValues] = useState<Record<string, number>>(emptyValues(VOICE_FEATURES));
  const [voiceResult, setVoiceResult] = useState<any | null>(null);
  const [voiceRunning, setVoiceRunning] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [voiceInfo, setVoiceInfo] = useState<any | null>(null);
  const [voiceInfoError, setVoiceInfoError] = useState<string | null>(null);
  const [showVoiceFields, setShowVoiceFields] = useState(false);

  // -- Risk/trend classifier state --
  const [riskValues, setRiskValues] = useState<Record<string, number>>({
    ...emptyValues(RISK_FEATURES),
    test_time: 0,
    baseline_total_UPDRS: 0,
  });
  const [riskResult, setRiskResult] = useState<any | null>(null);
  const [riskRunning, setRiskRunning] = useState(false);
  const [riskError, setRiskError] = useState<string | null>(null);
  const [riskInfo, setRiskInfo] = useState<any | null>(null);
  const [riskInfoError, setRiskInfoError] = useState<string | null>(null);

  useEffect(() => {
    const { token } = loadSession();
    if (!token) {
      router.push("/login");
      return;
    }
    api.mlClassifierInfo().then(setVoiceInfo).catch((e) => setVoiceInfoError(e.message));
    api.mlRiskClassifierInfo().then(setRiskInfo).catch((e) => setRiskInfoError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function loadSample(kind: "parkinsons" | "healthy") {
    const next: Record<string, number> = {};
    VOICE_FEATURES.forEach((f) => (next[f.alias] = f[kind]));
    setVoiceValues(next);
    setVoiceResult(null);
    setVoiceError(null);
  }

  async function runVoiceClassifier() {
    setVoiceRunning(true);
    setVoiceError(null);
    setVoiceResult(null);
    try {
      setVoiceResult(await api.mlClassify(voiceValues));
    } catch (e: any) {
      setVoiceError(e.message);
    } finally {
      setVoiceRunning(false);
    }
  }

  async function runRiskClassifier() {
    setRiskRunning(true);
    setRiskError(null);
    setRiskResult(null);
    try {
      setRiskResult(await api.mlRiskClassify(riskValues));
    } catch (e: any) {
      setRiskError(e.message);
    } finally {
      setRiskRunning(false);
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold mb-1">ML Lab — standalone research demonstrations</h1>
        <p className="text-sm text-neuro-muted max-w-2xl">
          Two models trained and cross-validated on public UCI Parkinson&apos;s research datasets, served here for
          transparency. Neither is connected to any patient&apos;s data or the live diagnostic pipeline — candidate
          conditions for real patients come from the agent/RAG layer on the Analysis tab, not from these models.
        </p>
      </div>

      {/* -------- Disease classifier -------- */}
      <div className="card space-y-4">
        <div className="flex items-start justify-between flex-wrap gap-2">
          <div>
            <h2 className="font-semibold">Voice-based Parkinson&apos;s classifier</h2>
            <p className="text-xs text-neuro-muted mt-1">
              22 acoustic features from a sustained-vowel voice recording &rarr; healthy vs. Parkinson&apos;s.
            </p>
          </div>
          {voiceInfo && (
            <span className="badge badge-info">
              LOSO ROC-AUC {voiceInfo.loso_metrics?.roc_auc?.toFixed(3)} · recall{" "}
              {voiceInfo.loso_metrics?.recall?.toFixed(3)}
            </span>
          )}
        </div>

        {voiceInfoError && (
          <div className="text-xs text-neuro-warn">
            Model unavailable: {voiceInfoError}. Train it with{" "}
            <code className="text-neuro-accent2">ml/training/train_parkinsons_classifier.py</code>.
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => loadSample("parkinsons")}
            className="text-xs border border-neuro-border rounded-lg px-3 py-1.5 hover:border-neuro-accent"
          >
            Load real Parkinson&apos;s-pattern sample
          </button>
          <button
            onClick={() => loadSample("healthy")}
            className="text-xs border border-neuro-border rounded-lg px-3 py-1.5 hover:border-neuro-accent"
          >
            Load real healthy-pattern sample
          </button>
          <button onClick={() => setShowVoiceFields((v) => !v)} className="text-xs text-neuro-accent px-3 py-1.5">
            {showVoiceFields ? "Hide" : "Show"} the 22 features {showVoiceFields ? "▲" : "▼"}
          </button>
        </div>
        <div className="text-[11px] text-neuro-muted -mt-2">
          Both samples are real rows from the UCI Parkinsons voice dataset, not invented values.
        </div>

        {showVoiceFields && (
          <FeatureGrid
            fields={VOICE_FEATURES}
            values={voiceValues}
            onChange={(alias, v) => setVoiceValues((prev) => ({ ...prev, [alias]: v }))}
          />
        )}

        <button
          onClick={runVoiceClassifier}
          disabled={voiceRunning || !!voiceInfoError}
          className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50"
        >
          {voiceRunning ? "Running…" : "Run classifier"}
        </button>

        {voiceError && <div className="text-neuro-critical text-sm">{voiceError}</div>}

        {voiceResult && (
          <div className="border-t border-neuro-border pt-4 space-y-3">
            <div className="flex items-center gap-2">
              <span className={`badge ${voiceResult.predicted_label === "parkinsons" ? "badge-high" : "badge-ok"}`}>
                predicted: {voiceResult.predicted_label}
              </span>
            </div>
            <ProbabilityBar label="Parkinson's pattern" value={voiceResult.probability_parkinsons} tone="critical" />
            <ProbabilityBar label="Healthy pattern" value={voiceResult.probability_healthy} tone="ok" />
            <p className="text-[11px] text-neuro-muted">{voiceResult.disclaimer}</p>
          </div>
        )}
      </div>

      {/* -------- Risk / worsening-trend classifier -------- */}
      <div className="card space-y-4">
        <div className="flex items-start justify-between flex-wrap gap-2">
          <div>
            <h2 className="font-semibold">Symptom-trend (worsening) classifier</h2>
            <p className="text-xs text-neuro-muted mt-1">
              Change in voice biomarkers from a patient&apos;s own baseline visit &rarr; likely trending worse or not.
            </p>
          </div>
          {riskInfo && <span className="badge badge-warn">pooled ROC-AUC {riskInfo.pooled_metrics?.roc_auc?.toFixed(3)} — modest, honestly reported</span>}
        </div>

        {riskInfoError && (
          <div className="text-xs text-neuro-warn">
            Model unavailable: {riskInfoError}. Train it with{" "}
            <code className="text-neuro-accent2">ml/training/train_risk_classifier.py</code>.
          </div>
        )}

        <div className="text-[11px] text-neuro-muted">
          All fields default to 0 (no change from baseline) — this model takes CHANGE values, not raw measurements.
          Adjust them to see how the prediction shifts; there is no pre-filled "real case" here, unlike the
          classifier above.
        </div>

        <FeatureGrid
          fields={RISK_FEATURES}
          values={riskValues}
          onChange={(alias, v) => setRiskValues((prev) => ({ ...prev, [alias]: v }))}
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Days since baseline visit</div>
            <input
              type="number"
              value={riskValues.test_time}
              onChange={(e) => setRiskValues((p) => ({ ...p, test_time: Number(e.target.value) }))}
              className="w-full bg-neuro-panel2 border border-neuro-border rounded-md px-2 py-1.5 text-xs"
            />
          </label>
          <label className="text-xs">
            <div className="text-neuro-muted mb-1">Baseline total_UPDRS</div>
            <input
              type="number"
              value={riskValues.baseline_total_UPDRS}
              onChange={(e) => setRiskValues((p) => ({ ...p, baseline_total_UPDRS: Number(e.target.value) }))}
              className="w-full bg-neuro-panel2 border border-neuro-border rounded-md px-2 py-1.5 text-xs"
            />
          </label>
        </div>

        <button
          onClick={runRiskClassifier}
          disabled={riskRunning || !!riskInfoError}
          className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50"
        >
          {riskRunning ? "Running…" : "Run classifier"}
        </button>

        {riskError && <div className="text-neuro-critical text-sm">{riskError}</div>}

        {riskResult && (
          <div className="border-t border-neuro-border pt-4 space-y-3">
            <div className="flex items-center gap-2">
              <span className={`badge ${riskResult.predicted_label === "worsening" ? "badge-high" : "badge-ok"}`}>
                predicted: {riskResult.predicted_label.replace(/_/g, " ")}
              </span>
            </div>
            <ProbabilityBar label="Worsening" value={riskResult.probability_worsening} tone="critical" />
            <ProbabilityBar label="Stable or improving" value={riskResult.probability_stable_or_improving} tone="ok" />
            <p className="text-[11px] text-neuro-muted">{riskResult.disclaimer}</p>
          </div>
        )}
      </div>

      <div className="text-[11px] text-neuro-muted">
        Both models were validated with subject-level cross-validation (no patient&apos;s data appears in both train
        and test) after an earlier row-level split was found to leak subjects between the two — see the classifier
        card&apos;s reported metrics above for the honest, leakage-free numbers.
      </div>
    </div>
  );
}
