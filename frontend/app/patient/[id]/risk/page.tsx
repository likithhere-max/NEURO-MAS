"use client";

import Link from "next/link";
import { usePatient } from "@/components/PatientContext";
import RiskBadge from "@/components/RiskBadge";

// Deterministic, non-LLM urgency copy keyed off the SAME heuristic bucket
// the backend risk agent already computes (app/agents/risk.py `_bucket()`).
// This mapping never asks the model to phrase urgency -- it's a fixed
// lookup, matching section 8's "the risk engine must remain deterministic".
const URGENCY_TEXT: Record<string, string> = {
  low: "Routine follow-up as usual — no urgent action indicated by current heuristics.",
  medium: "Consider scheduling a follow-up with a clinician in the near term.",
  high: "Prompt clinical evaluation is recommended.",
  critical: "Seek immediate medical evaluation.",
};

const FACTOR_LABELS: Record<string, string> = {
  fall_risk: "Fall risk",
  seizure_risk: "Seizure risk",
  deterioration_risk: "Deterioration risk",
};

export default function RiskPage() {
  const { patientId, latestWorkflow } = usePatient();

  if (!latestWorkflow) {
    return (
      <div className="card text-center py-12">
        <div className="text-lg font-semibold mb-2">No risk assessment yet</div>
        <Link
          href={`/patient/${patientId}/assessment`}
          className="inline-block bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm"
        >
          Start Symptom Assessment
        </Link>
      </div>
    );
  }

  const riskResult = latestWorkflow.agent_results?.risk_stratification_agent || {};
  const risk = riskResult.raw_payload || {};
  const diagnostic = latestWorkflow.agent_results?.neuro_diagnostic_agent?.raw_payload || {};
  const dataIntake = latestWorkflow.agent_results?.data_intake_agent?.raw_payload || {};

  const elevatedFactors = Object.entries(risk.heuristic_scores || {}).filter(([, v]: any) => v > 0);
  const uncertainFactors = [
    ...(diagnostic.missing_information || []),
    ...(dataIntake.missing_data || []).map((m: string) => `missing data: ${m.replaceAll("_", " ")}`),
  ];

  return (
    <div className="max-w-3xl space-y-6">
      <div className="card text-center py-8">
        <div className="text-xs text-neuro-muted mb-2">Current risk level</div>
        <RiskBadge bucket={risk.overall_bucket} size="lg" />
        <p className="text-sm text-neuro-muted mt-3 max-w-md mx-auto">
          Heuristic score, not a validated clinical risk score — for clinician review.
        </p>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Main contributing factors</h2>
        {elevatedFactors.length === 0 && <div className="text-sm text-neuro-muted">No elevated risk factors detected.</div>}
        <div className="space-y-3">
          {elevatedFactors.map(([name, value]: any) => (
            <div key={name}>
              <div className="flex justify-between text-sm mb-1">
                <span>{FACTOR_LABELS[name] || name}</span>
                <span className="text-neuro-muted">{Math.round(value * 100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-neuro-panel2 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-neuro-accent to-neuro-high" style={{ width: `${Math.round(value * 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">Protective / uncertain factors</h2>
        {uncertainFactors.length === 0 ? (
          <div className="text-sm text-neuro-muted">No missing information flagged.</div>
        ) : (
          <ul className="list-disc list-inside text-sm space-y-1 text-neuro-muted">
            {uncertainFactors.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">Why this matters</h2>
        <p className="text-sm text-neuro-text">{riskResult.reasoning_summary || "No explanation available."}</p>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">Recommended urgency</h2>
        <p className="text-sm text-neuro-text">{URGENCY_TEXT[risk.overall_bucket] || "Unknown."}</p>
        <div className="text-[11px] text-neuro-muted mt-2">
          This is general decision-support guidance, not a substitute for professional medical evaluation.
        </div>
      </div>
    </div>
  );
}
