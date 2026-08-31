"use client";

import Link from "next/link";
import { usePatient } from "@/components/PatientContext";

export default function HistoryPage() {
  const { patientId, assessments } = usePatient();
  const newestWorkflowId = assessments[0]?.workflow_id;

  return (
    <div className="max-w-3xl space-y-4">
      <h1 className="font-semibold text-lg">Assessment history</h1>
      {assessments.length === 0 && <div className="text-sm text-neuro-muted">No assessments yet.</div>}
      <div className="space-y-3">
        {assessments.map((a: any, i: number) => (
          <div key={a.id || i} className="card">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-neuro-muted">{new Date(a.created_at).toLocaleString()}</span>
              {a.workflow_id && a.workflow_id === newestWorkflowId && (
                <Link href={`/patient/${patientId}/analysis`} className="text-xs text-neuro-accent">
                  view analysis &rarr;
                </Link>
              )}
            </div>
            <div className="text-sm">{a.free_text}</div>
            {a.extracted_json?.symptoms?.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {a.extracted_json.symptoms.map((s: any, j: number) => (
                  <span key={j} className="badge badge-info">
                    {s.name}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
