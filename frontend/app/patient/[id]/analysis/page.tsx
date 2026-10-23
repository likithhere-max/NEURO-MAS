"use client";

import { useState } from "react";
import Link from "next/link";
import { usePatient } from "@/components/PatientContext";

export default function AnalysisPage() {
  const { patientId, latestAssessment, latestWorkflow } = usePatient();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [explainLevel, setExplainLevel] = useState<"simple" | "detailed" | "technical">("simple");

  if (!latestWorkflow) {
    return (
      <div className="card text-center py-12">
        <div className="text-lg font-semibold mb-2">No analysis yet</div>
        <Link
          href={`/patient/${patientId}/assessment`}
          className="inline-block bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm"
        >
          Start Symptom Assessment
        </Link>
      </div>
    );
  }

  const diagnostic = latestWorkflow.agent_results?.neuro_diagnostic_agent || {};
  const diagPayload = diagnostic.raw_payload || {};
  const candidates = diagPayload.candidate_conditions || [];
  const risk = latestWorkflow.agent_results?.risk_stratification_agent?.raw_payload || {};
  const knowledge = latestWorkflow.agent_results?.knowledge_agent?.raw_payload || {};
  const extracted = latestAssessment?.extracted_json || {};

  return (
    <div className="space-y-6">
      <div className="card">
        <h1 className="font-semibold text-lg mb-1">Possible neurological conditions considered</h1>
        <p className="text-xs text-neuro-muted mb-4">
          These are AI-generated candidate conditions for clinician review — not a confirmed diagnosis.
        </p>
        <div className="space-y-3">
          {candidates.map((c: any, i: number) => (
            <div key={c.disease_id} className="border border-neuro-border rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-neuro-muted text-sm">{i + 1}.</span>
                  <Link href={`/patient/${patientId}/analysis/${c.disease_id}`} className="font-medium hover:text-neuro-accent">
                    {c.preferred_name}
                  </Link>
                </div>
                <span className="badge badge-info">{Math.round((c.confidence || 0) * 100)}% match</span>
              </div>
              <div className="h-1.5 rounded-full bg-neuro-panel2 overflow-hidden mt-2">
                <div
                  className="h-full bg-gradient-to-r from-neuro-accent to-neuro-accent2"
                  style={{ width: `${Math.round((c.confidence || 0) * 100)}%` }}
                />
              </div>
              <button onClick={() => setExpanded(expanded === c.disease_id ? null : c.disease_id)} className="text-xs text-neuro-accent mt-2">
                {expanded === c.disease_id ? "Hide" : "Why was this considered?"}
              </button>
              {expanded === c.disease_id && (
                <ul className="text-xs text-neuro-muted mt-2 list-disc list-inside space-y-1">
                  {(c.matched_terms || []).length > 0 && <li>Matched symptom terms: {c.matched_terms.join(", ")}</li>}
                  <li>
                    Retrieval method:{" "}
                    {c.retrieval_method === "semantic_embedding" ? "semantic similarity search" : "keyword overlap (fallback)"}
                  </li>
                  {extracted.duration && <li>Reported duration: {extracted.duration}</li>}
                  {extracted.progression && <li>Progression: {extracted.progression}</li>}
                </ul>
              )}
            </div>
          ))}
          {candidates.length === 0 && <div className="text-sm text-neuro-muted">No candidate conditions matched the reported symptoms.</div>}
        </div>
        {diagPayload.missing_information?.length > 0 && (
          <div className="text-xs text-neuro-warn mt-3">⚠ {diagPayload.missing_information.join(" ")}</div>
        )}
        {diagPayload.red_flags?.length > 0 && (
          <div className="text-xs text-neuro-warn mt-2">
            Warning signs to check for in these conditions (a general checklist from the knowledge base, not findings
            detected in this patient): {diagPayload.red_flags.join(", ")}
          </div>
        )}
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h2 className="font-semibold">Explain Result</h2>
          <div className="flex gap-1 text-xs">
            {(["simple", "detailed", "technical"] as const).map((l) => (
              <button
                key={l}
                onClick={() => setExplainLevel(l)}
                className={`px-2 py-1 rounded-md border capitalize ${
                  explainLevel === l ? "border-neuro-accent text-neuro-accent" : "border-neuro-border text-neuro-muted"
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        {explainLevel === "simple" && <p className="text-sm text-neuro-text">{diagnostic.reasoning_summary || "No explanation available."}</p>}

        {explainLevel === "detailed" && (
          <div className="text-sm space-y-2">
            <div>
              <span className="font-medium">Symptom contribution: </span>
              {(extracted.symptoms || []).map((s: any) => s.name).join(", ") || "none extracted"}
            </div>
            <div>
              <span className="font-medium">Clinical feature contribution: </span>
              {(extracted.associated_features || []).join(", ") || "none"}, duration {extracted.duration || "unspecified"}, progression{" "}
              {extracted.progression || "unspecified"}
            </div>
            <div>
              <span className="font-medium">Model contribution: </span>
              {candidates.length} candidate(s) via{" "}
              {candidates[0]?.retrieval_method === "semantic_embedding" ? "semantic retrieval" : "lexical fallback"}, confidence capped at
              0.75 (heuristic, not a validated classifier)
            </div>
            <div>
              <span className="font-medium">Knowledge contribution: </span>
              {knowledge.citations?.length || 0} evidence citation(s) retrieved
            </div>
            <div>
              <span className="font-medium">Risk contribution: </span>overall bucket &quot;{risk.overall_bucket}&quot;
            </div>
            <div>
              <span className="font-medium">Uncertainty: </span>
              {diagPayload.missing_information?.join("; ") || "none flagged"}
            </div>
          </div>
        )}

        {explainLevel === "technical" && (
          <pre className="text-xs text-neuro-muted whitespace-pre-wrap overflow-x-auto max-h-96 overflow-y-auto">
            {JSON.stringify({ diagnostic: diagPayload, risk, knowledge, extracted }, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
