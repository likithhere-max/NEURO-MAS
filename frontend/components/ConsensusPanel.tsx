"use client";

import { formatSupport, hypothesisStatusLabel } from "@/lib/labels";

export default function ConsensusPanel({ consensus }: { consensus: any }) {
  if (!consensus) {
    return (
      <div className="card">
        <h2 className="font-semibold mb-2">Consensus Panel</h2>
        <div className="text-sm text-neuro-muted">No competing hypotheses were generated for this run.</div>
      </div>
    );
  }
  const hyps = consensus.hypotheses || [];
  return (
    <div className="card">
      <h2 className="font-semibold mb-1">Consensus Panel</h2>
      <div className="text-xs text-neuro-muted mb-3">
        {consensus.calibrated
          ? "Calibrated consensus score."
          : "Uncalibrated support index over independent evidence groups — not a probability and not a validated clinical score. Only evidence that names a specific condition counts; other agents are listed as non-voting."}
      </div>
      <div className="space-y-2 mb-3">
        {hyps.map((h: any) => (
          <div key={h.hypothesis}>
            <div className="flex justify-between text-sm mb-1">
              <span>{h.hypothesis}</span>
              <span className="text-neuro-muted">
                {formatSupport(h, consensus.calibrated)}
                {hypothesisStatusLabel(h.status) ? ` · ${hypothesisStatusLabel(h.status)}` : ""}
              </span>
            </div>
            <div className="h-2 rounded-full bg-neuro-panel2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-neuro-accent to-neuro-accent2"
                style={{ width: `${Math.round(Math.min(1, Math.max(0, h.consensus_score || 0)) * 100)}%` }}
              />
            </div>
            <div className="text-[11px] text-neuro-muted mt-1">
              voters: {h.contributing_agents?.join(", ") || "none"}
              {h.abstained_agents?.length ? ` · abstained: ${h.abstained_agents.join(", ")}` : ""}
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-between border-t border-neuro-border pt-3 text-sm">
        <div>
          Disagreement: <span className="font-semibold">{Math.round((consensus.disagreement || 0) * 100)}%</span>
          {consensus.margin != null && (
            <span className="text-neuro-muted"> · lead over runner-up {Math.round(consensus.margin * 100)} pts</span>
          )}
        </div>
        <span className={`badge ${consensus.action_permitted ? "badge-ok" : "badge-warn"}`}>
          {consensus.action_permitted ? "autonomous action permitted" : "additional analysis required"}
        </span>
      </div>
      <div className="text-xs text-neuro-muted mt-2">{consensus.reason}</div>
      {consensus.ties?.length > 0 && (
        <div className="text-xs text-neuro-warn mt-1">No single leader: tied within tolerance — {consensus.ties.join(", ")}.</div>
      )}
      {consensus.caveats?.length > 0 && (
        <ul className="text-[11px] text-neuro-muted mt-2 list-disc pl-4">
          {consensus.caveats.map((c: string) => <li key={c}>{c}</li>)}
        </ul>
      )}
    </div>
  );
}
