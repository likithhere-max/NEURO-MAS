"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "@/lib/api";
import { usePatient } from "@/components/PatientContext";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <div className="text-sm font-medium mb-1">{title}</div>
      <div>{children}</div>
    </div>
  );
}

export default function DiseaseInfoPage() {
  const params = useParams();
  const patientId = params.id as string;
  const diseaseId = params.diseaseId as string;
  const { latestAssessment, latestWorkflow } = usePatient();
  const [disease, setDisease] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDisease(diseaseId).then(setDisease).catch((e) => setError(e.message));
  }, [diseaseId]);

  if (error) {
    return (
      <div className="max-w-3xl">
        <Link href={`/patient/${patientId}/analysis`} className="text-xs text-neuro-muted">
          &larr; back to analysis
        </Link>
        <div className="text-neuro-critical text-sm mt-3">{error}</div>
      </div>
    );
  }
  if (!disease) return <div className="text-sm text-neuro-muted">Loading…</div>;

  const reportedNames: string[] = ((latestAssessment?.extracted_json?.symptoms) || []).map((s: any) => s.name.toLowerCase());
  const reportedFeatures: string[] = (latestAssessment?.extracted_json?.associated_features || []).map((f: string) => f.toLowerCase());
  const reported = [...reportedNames, ...reportedFeatures];
  const observed = (disease.symptoms || []).filter((s: string) =>
    reported.some((r) => s.toLowerCase().includes(r) || r.includes(s.toLowerCase().split(" ")[0]))
  );
  const notObserved = (disease.symptoms || []).filter((s: string) => !observed.includes(s));

  const candidate = (latestWorkflow?.agent_results?.neuro_diagnostic_agent?.raw_payload?.candidate_conditions || []).find(
    (c: any) => c.disease_id === diseaseId
  );

  return (
    <div className="max-w-3xl space-y-4">
      <Link href={`/patient/${patientId}/analysis`} className="text-xs text-neuro-muted">
        &larr; back to analysis
      </Link>
      <div className="card">
        <h1 className="text-lg font-semibold">{disease.preferred_name}</h1>
        <div className="text-xs text-neuro-muted mb-3">
          {disease.parent_category}
          {disease.sub_category ? ` · ${disease.sub_category}` : ""}
        </div>
        <div className="rounded-lg bg-neuro-panel2 border border-neuro-border px-3 py-2 text-xs text-neuro-muted mb-4">
          The system identified this as a condition worth considering based on the available information — this is
          not a confirmed diagnosis.
          {candidate && ` Match confidence: ${Math.round((candidate.confidence || 0) * 100)}%.`}
        </div>

        <Section title="What it is">{disease.definition}</Section>

        <Section title="Common symptoms">
          <ul className="list-disc list-inside text-sm space-y-1">
            {(disease.symptoms || []).map((s: string) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </Section>

        <Section title="Symptoms observed in this patient">
          {observed.length ? (
            <ul className="list-disc list-inside text-sm space-y-1 text-neuro-ok">
              {observed.map((s: string) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          ) : (
            <span className="text-sm text-neuro-muted">
              None of the textbook symptom names were directly matched — see the candidate&apos;s matched terms in Analysis.
            </span>
          )}
        </Section>

        <Section title="Symptoms not observed / unknown">
          <ul className="list-disc list-inside text-sm space-y-1 text-neuro-muted">
            {notObserved.map((s: string) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </Section>

        <Section title="Why it was considered">
          {candidate
            ? `Matched terms: ${(candidate.matched_terms || []).join(", ") || "semantic similarity to reported symptoms"}.`
            : "Not part of this patient's current candidate list."}
        </Section>

        <Section title="Typical clinical evaluation">
          <ul className="list-disc list-inside text-sm space-y-1">
            {(disease.diagnostic_methods || []).map((s: string) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </Section>

        <Section title="General management categories">
          <div className="text-sm text-neuro-muted">{disease.treatment_status}</div>
        </Section>

        <Section title="When to seek urgent care">
          <div className="text-sm text-neuro-critical">{disease.emergency_criteria}</div>
          {disease.red_flags?.length > 0 && (
            <ul className="list-disc list-inside text-sm mt-1 space-y-1">
              {disease.red_flags.map((s: string) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          )}
        </Section>

        <Section title="Evidence sources">
          <ul className="text-xs space-y-1">
            {(disease.source_urls || []).map((u: string) => (
              <li key={u}>
                <a href={u} target="_blank" rel="noreferrer" className="text-neuro-accent underline break-all">
                  {u}
                </a>
              </li>
            ))}
          </ul>
          <div className="text-[11px] text-neuro-muted mt-1">
            Last verified: {disease.last_verified} — curated prototype summary, pending clinician verification.
          </div>
        </Section>
      </div>
    </div>
  );
}
