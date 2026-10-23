"use client";

import { useState } from "react";
import Link from "next/link";
import { usePatient } from "@/components/PatientContext";
import { api } from "@/lib/api";

export default function EvidencePage() {
  const { patientId, latestWorkflow } = usePatient();
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  async function search() {
    if (!query.trim()) return;
    setSearching(true);
    setSearchError(null);
    try {
      const res = await api.searchKnowledge(query);
      setSearchResults(res.results || []);
    } catch (e: any) {
      setSearchResults(null);
      setSearchError("Search failed. Please check your connection and try again.");
    } finally {
      setSearching(false);
    }
  }

  const knowledge = latestWorkflow?.agent_results?.knowledge_agent || {};
  const citations = knowledge.raw_payload?.citations || [];

  return (
    <div className="space-y-6">
      <div className="card">
        <h1 className="font-semibold text-lg mb-1">Supporting evidence</h1>
        <p className="text-xs text-neuro-muted mb-4">
          Source-backed evidence retrieved for this assessment&apos;s candidate conditions. Nothing shown here is
          fabricated — if no matching entry exists in the knowledge base, no citation is invented to fill the gap.
        </p>
        {citations.length === 0 && (
          <div className="text-sm text-neuro-muted">
            {latestWorkflow ? "No matching knowledge-base entry for the current candidates." : "No assessment yet."}
          </div>
        )}
        <div className="space-y-3">
          {citations.map((c: any, i: number) => (
            <div key={i} className="border border-neuro-border rounded-lg p-3">
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">{c.disease}</span>
                <span className="badge badge-info">knowledge base</span>
              </div>
              <a href={c.source_url} target="_blank" rel="noreferrer" className="text-xs text-neuro-accent underline break-all">
                {c.source_url}
              </a>
              <div className="text-[11px] text-neuro-muted mt-1">
                Last verified {c.last_verified ?? "unknown"} — {c.review_status ?? "review status unknown"}
              </div>
            </div>
          ))}
        </div>
        {knowledge.reasoning_summary && (
          <div className="text-sm text-neuro-text mt-4 border-t border-neuro-border pt-3">{knowledge.reasoning_summary}</div>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">Explore the knowledge base</h2>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search symptoms or conditions…"
            className="flex-1 bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            onKeyDown={(e) => e.key === "Enter" && search()}
          />
          <button onClick={search} disabled={searching} className="text-sm border border-neuro-border rounded-lg px-4 py-2">
            {searching ? "Searching…" : "Search"}
          </button>
        </div>
        {searchError && <div className="mt-3 text-sm text-neuro-critical">{searchError}</div>}
        {searchResults && (
          <div className="mt-3 space-y-2">
            {searchResults.length === 0 && <div className="text-sm text-neuro-muted">No matches.</div>}
            {searchResults.map((d: any) => (
              <Link
                key={d.disease_id}
                href={`/patient/${patientId}/analysis/${d.disease_id}`}
                className="block border border-neuro-border rounded-lg px-3 py-2 hover:border-neuro-accent"
              >
                <div className="text-sm font-medium">{d.preferred_name}</div>
                <div className="text-xs text-neuro-muted">{(d.definition || "").slice(0, 140)}…</div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
