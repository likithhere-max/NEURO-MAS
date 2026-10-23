"use client";

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
        Weighted consensus over agent hypotheses — not a validated clinical
        scoring formula and not a disease probability. Only agents that give a
        score for a specific condition vote; the rest abstain (listed below).
      </div>
      <div className="space-y-2 mb-3">
        {hyps.map((h: any) => (
          <div key={h.hypothesis}>
            <div className="flex justify-between text-sm mb-1">
              <span>{h.hypothesis}</span>
              <span className="text-neuro-muted">{Math.round(h.consensus_score * 100)}%</span>
            </div>
            <div className="h-2 rounded-full bg-neuro-panel2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-neuro-accent to-neuro-accent2"
                style={{ width: `${Math.round(h.consensus_score * 100)}%` }}
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
    </div>
  );
}
