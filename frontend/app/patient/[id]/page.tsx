"use client";

import { useState } from "react";
import Link from "next/link";
import { usePatient } from "@/components/PatientContext";
import { loadSession } from "@/lib/api";
import RiskBadge from "@/components/RiskBadge";
import AgentActivityPanel from "@/components/AgentActivityPanel";
import ConsensusPanel from "@/components/ConsensusPanel";
import PersonalizationPanel from "@/components/PersonalizationPanel";
import ActionsPanel from "@/components/ActionsPanel";
import AlertCenter from "@/components/AlertCenter";
import TimelinePanel from "@/components/TimelinePanel";

export default function PatientDashboard() {
  const { patientId, latestAssessment, latestWorkflow, alerts, timeline, refreshAll, loadingWorkflow } = usePatient();
  const [showPipeline, setShowPipeline] = useState(false);
  const { role } = loadSession();

  if (loadingWorkflow) {
    return <div className="text-sm text-neuro-muted">Loading assessment…</div>;
  }

  if (!latestWorkflow) {
    return (
      <div className="card text-center py-12">
        <div className="text-lg font-semibold mb-2">No assessment yet</div>
        <p className="text-sm text-neuro-muted mb-4 max-w-md mx-auto">
          Describe symptoms in the patient&apos;s own words to run the full neurological multi-agent
          assessment — candidate conditions, risk, evidence, and next steps.
        </p>
        <Link
          href={`/patient/${patientId}/assessment`}
          className="inline-block bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm"
        >
          Start Symptom Assessment
        </Link>
      </div>
    );
  }

  const diagnostic = latestWorkflow.agent_results?.neuro_diagnostic_agent?.raw_payload || {};
  const risk = latestWorkflow.agent_results?.risk_stratification_agent?.raw_payload || {};
  const personalization = latestWorkflow.agent_results?.personalization_agent?.raw_payload || {};
  const topCandidates = (diagnostic.candidate_conditions || []).slice(0, 3);
  const extracted = latestAssessment?.extracted_json || {};

  return (
    <div className="space-y-6">
      {latestWorkflow.is_emergency && (
        <div className="rounded-lg border border-neuro-critical bg-neuro-critical/10 px-4 py-3 text-sm text-neuro-critical font-medium">
          ⚠ Deterministic emergency detector triggered on this assessment — independent of agent consensus. If this
          reflects a real emergency, seek immediate care.
        </div>
      )}

      {latestWorkflow.persisted === false && (
        <div className="rounded-lg border border-neuro-critical bg-neuro-critical/10 px-4 py-3 text-sm text-neuro-critical">
          This result could not be saved to the record. Do not rely on it appearing in history; repeat the assessment later.
        </div>
      )}
      {latestWorkflow.degraded && (
        <div className="rounded-lg border border-neuro-warn bg-neuro-warn/10 px-4 py-3 text-sm">
          Incomplete analysis: {(latestWorkflow.failed_agents || []).join(", ")} did not complete. Treat the findings below as partial.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="text-xs text-neuro-muted mb-1">Main symptoms reported</div>
          <div className="flex flex-wrap gap-1">
            {(extracted.symptoms || []).map((s: any, i: number) => (
              <span key={i} className="badge badge-info">
                {s.name}
              </span>
            ))}
            {(extracted.symptoms || []).length === 0 && <span className="text-sm text-neuro-muted">none extracted</span>}
          </div>
        </div>
        <div className="card">
          <div className="text-xs text-neuro-muted mb-1">Current risk</div>
          <RiskBadge bucket={risk.overall_bucket} size="lg" />
          <Link href={`/patient/${patientId}/risk`} className="block text-xs text-neuro-accent mt-2">
            Why? &rarr;
          </Link>
        </div>
        <div className="card">
          <div className="text-xs text-neuro-muted mb-1">Last updated</div>
          <div className="text-sm">{new Date(latestWorkflow.started_at).toLocaleString()}</div>
          <Link href={`/patient/${patientId}/assessment`} className="block text-xs text-neuro-accent mt-2">
            Update assessment &rarr;
          </Link>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">AI assessment — candidate conditions considered</h2>
          <Link href={`/patient/${patientId}/analysis`} className="text-xs text-neuro-accent">
            Full analysis &rarr;
          </Link>
        </div>
        {topCandidates.length === 0 && <div className="text-sm text-neuro-muted">No candidate conditions matched.</div>}
        <div className="space-y-2">
          {topCandidates.map((c: any) => (
            <div key={c.disease_id} className="flex items-center justify-between border border-neuro-border rounded-lg px-3 py-2">
              <span className="text-sm">{c.preferred_name}</span>
              <span className="text-xs text-neuro-muted">{Math.round((c.confidence || 0) * 100)}% match</span>
            </div>
          ))}
        </div>
        <div className="text-[11px] text-neuro-muted mt-3">
          AI-generated candidate conditions for clinician review — not a confirmed diagnosis.
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="font-semibold mb-2">Recommended next action</h2>
          <p className="text-sm text-neuro-text">
            {latestWorkflow.is_emergency
              ? "Seek immediate emergency medical care."
              : personalization.monitoring_plan?.[0] || "No specific action suggested yet."}
          </p>
          <Link href={`/patient/${patientId}/next-steps`} className="block text-xs text-neuro-accent mt-2">
            See full next steps &rarr;
          </Link>
        </div>
        <div className="card">
          <h2 className="font-semibold mb-2">Ask the AI Assistant</h2>
          <p className="text-sm text-neuro-muted mb-3">Ask about your assessment, risk, or what to do next.</p>
          <Link
            href={`/patient/${patientId}/chat`}
            className="inline-block bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm"
          >
            Open AI Assistant
          </Link>
        </div>
      </div>

      <div>
        <button onClick={() => setShowPipeline((v) => !v)} className="text-sm text-neuro-accent mb-3">
          {showPipeline ? "Hide" : "Show"} analysis pipeline (technical detail) {showPipeline ? "▲" : "▼"}
        </button>
        {showPipeline && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <AgentActivityPanel agentResults={latestWorkflow.agent_results || {}} />
              <ConsensusPanel consensus={latestWorkflow.consensus} />
              <PersonalizationPanel agentResults={latestWorkflow.agent_results || {}} />
            </div>
            <div className="space-y-6">
              <ActionsPanel actions={latestWorkflow.proposed_actions || []} role={role} onChanged={refreshAll} />
              <AlertCenter alerts={alerts} onChanged={refreshAll} />
              <TimelinePanel events={latestWorkflow.audit_trail || timeline} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
