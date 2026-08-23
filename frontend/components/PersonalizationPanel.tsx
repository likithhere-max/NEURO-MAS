"use client";

export default function PersonalizationPanel({ agentResults }: { agentResults: Record<string, any> }) {
  const p = agentResults?.personalization_agent;
  if (!p) {
    return (
      <div className="card">
        <h2 className="font-semibold mb-2">Personalization</h2>
        <div className="text-sm text-neuro-muted">No personalization output yet.</div>
      </div>
    );
  }
  const plan = p.raw_payload?.monitoring_plan || [];
  const reasons = p.raw_payload?.personalization_reasons || [];
  const education = p.raw_payload?.educational_content || [];
  return (
    <div className="card">
      <h2 className="font-semibold mb-1">Personalization</h2>
      <div className="text-xs text-neuro-muted mb-3">Why this is specific to this patient, not a generic template.</div>
      <div className="mb-3">
        <div className="text-sm font-medium mb-1">Monitoring plan</div>
        <ul className="text-sm space-y-1 list-disc list-inside text-neuro-text">
          {plan.map((item: string, i: number) => <li key={i}>{item}</li>)}
        </ul>
      </div>
      {education.length > 0 && (
        <div className="mb-3">
          <div className="text-sm font-medium mb-1">Educational content</div>
          <ul className="text-sm space-y-1 list-disc list-inside text-neuro-text">
            {education.map((item: string, i: number) => <li key={i}>{item}</li>)}
          </ul>
        </div>
      )}
      {reasons.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-1">Basis for personalization</div>
          <ul className="text-xs space-y-1 list-disc list-inside text-neuro-muted">
            {reasons.map((item: string, i: number) => <li key={i}>{item}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
