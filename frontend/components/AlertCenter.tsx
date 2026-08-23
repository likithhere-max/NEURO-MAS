"use client";

import { api } from "@/lib/api";

const SEV_BADGE: Record<string, string> = {
  low: "badge-ok",
  medium: "badge-warn",
  high: "badge-high",
  critical: "badge-critical",
};

export default function AlertCenter({ alerts, onChanged }: { alerts: any[]; onChanged: () => void }) {
  async function ack(id: string) {
    await api.acknowledgeAlert(id);
    onChanged();
  }
  return (
    <div className="card">
      <h2 className="font-semibold mb-3">Alert Center</h2>
      {(!alerts || alerts.length === 0) && <div className="text-sm text-neuro-muted">No alerts for this patient.</div>}
      <div className="space-y-2">
        {alerts?.map((a) => (
          <div key={a.id} className="border border-neuro-border rounded-lg px-3 py-2">
            <div className="flex items-center justify-between">
              <span className={`badge ${SEV_BADGE[a.severity] || "badge-info"}`}>{a.severity}</span>
              <span className="text-xs text-neuro-muted">{a.status}</span>
            </div>
            <div className="text-sm mt-1">{a.reason}</div>
            {a.status === "open" && (
              <button onClick={() => ack(a.id)} className="mt-2 text-xs border border-neuro-border rounded-md px-3 py-1">
                Acknowledge
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
