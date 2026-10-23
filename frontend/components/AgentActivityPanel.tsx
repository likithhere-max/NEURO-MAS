"use client";

const STATUS_STYLE: Record<string, string> = {
  completed: "badge-ok",
  skipped: "badge-warn",
  failed: "badge-critical",
  model_not_available: "badge-warn",
  running: "badge-info",
  pending: "badge-info",
};

const STATUS_ICON: Record<string, string> = {
  completed: "✓",
  skipped: "–",
  failed: "✕",
  model_not_available: "⚠",
  running: "→",
  pending: "…",
};

const AGENT_LABELS: Record<string, string> = {
  data_intake_agent: "Data Intake Agent",
  emergency_escalation_agent: "Emergency / Escalation Agent",
  neuro_diagnostic_agent: "Neuro-Diagnostic Agent",
  signal_analysis_agent: "Signal Analysis Agent",
  longitudinal_agent: "Longitudinal / Biomarker Agent",
  medication_safety_agent: "Medication Safety Agent",
  knowledge_agent: "Knowledge / RAG Agent",
  risk_stratification_agent: "Risk Stratification Agent",
  personalization_agent: "Personalization Agent",
  patient_copilot_agent: "Patient Copilot Agent",
};

export default function AgentActivityPanel({ agentResults }: { agentResults: Record<string, any> }) {
  const names = Object.keys(agentResults);
  return (
    <div className="card">
      <h2 className="font-semibold mb-3">Agent Activity</h2>
      {names.length === 0 && <div className="text-sm text-neuro-muted">No workflow run yet.</div>}
      <div className="space-y-2">
        {names.map((name) => {
          const r = agentResults[name];
          const style = STATUS_STYLE[r.status] || "badge-info";
          return (
            <div key={name} className="flex items-center justify-between border border-neuro-border rounded-lg px-3 py-2">
              <div>
                <div className="text-sm font-medium">{AGENT_LABELS[name] || name}</div>
                <div className="text-xs text-neuro-muted max-w-md">{r.reasoning_summary}</div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {(r.warnings || []).some((w: string) => w.startsWith("confidence_is_")) ? (
                  <span className="text-xs text-neuro-muted" title="Input/rule-table coverage. NOT a probability of being correct.">
                    coverage {Math.round((r.confidence || 0) * 100)}%
                  </span>
                ) : (
                  <span className="text-xs text-neuro-muted" title="Heuristic agent score. NOT a calibrated probability.">
                    score {Math.round((r.confidence || 0) * 100)}%
                  </span>
                )}
                <span className={`badge ${style}`}>
                  {STATUS_ICON[r.status] || "•"} {r.status}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
