"use client";

import { useState } from "react";
import { api } from "@/lib/api";

const RISK_BADGE: Record<string, string> = {
  none: "badge-ok",
  low: "badge-ok",
  medium: "badge-warn",
  high: "badge-high",
  restricted: "badge-critical",
};

// "executed" is only shown when the backend confirmed delivery or a clinician approved. A permitted action
// with no notification channel configured stays "proposed" -- say so instead of implying it was sent.
const STATUS_LABEL: Record<string, string> = {
  proposed: "proposed · not sent",
  pending_approval: "awaiting clinician approval",
  executed: "executed",
  rejected: "rejected",
};

export default function ActionsPanel({ actions, role, onChanged }: { actions: any[]; role: string | null; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null);
  async function approve(actionId: string) {
    setError(null);
    try {
      await api.approveAction(actionId);
      onChanged();
    } catch (e: any) {
      setError("Could not approve this action. " + (e?.message?.startsWith("403") ? "Only clinicians can approve." : "Please retry."));
    }
  }

  return (
    <div className="card">
      <h2 className="font-semibold mb-1">Autonomous Action Engine</h2>
      <div className="text-xs text-neuro-muted mb-3">
        Every action is tagged with an autonomy level (0–5). Restricted actions
        (medication changes, diagnosis confirmation) can never be auto-approved —
        only a clinician can approve them here.
      </div>
      {error && <div className="text-xs text-neuro-critical mb-2">{error}</div>}
      {(!actions || actions.length === 0) && <div className="text-sm text-neuro-muted">No actions proposed yet.</div>}
      <div className="space-y-2">
        {actions?.map((a) => (
          <div key={a.action_id} className="border border-neuro-border rounded-lg px-3 py-2">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">{a.action_type.replaceAll("_", " ")}</div>
              <div className="flex items-center gap-2">
                <span className={`badge ${RISK_BADGE[a.risk_level] || "badge-info"}`}>L{a.autonomy_level} · {a.risk_level}</span>
                <span className="badge badge-info">{STATUS_LABEL[a.status] || a.status.replaceAll("_", " ")}</span>
              </div>
            </div>
            <div className="text-xs text-neuro-muted mt-1">{a.reason}</div>
            {a.requires_human_approval && a.status === "pending_approval" && role === "clinician" && (
              <button
                onClick={() => approve(a.action_id)}
                className="mt-2 text-xs bg-neuro-accent text-neuro-bg font-medium rounded-md px-3 py-1"
              >
                Approve as clinician
              </button>
            )}
            {a.requires_human_approval && a.status === "pending_approval" && role !== "clinician" && (
              <div className="mt-2 text-xs text-neuro-muted italic">Awaiting clinician authorization</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
