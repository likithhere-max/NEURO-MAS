"use client";

import { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { loadSession } from "@/lib/api";
import { PatientProvider, usePatient } from "@/components/PatientContext";
import TabNav from "@/components/TabNav";
import RiskBadge from "@/components/RiskBadge";

function PatientHeader() {
  const { patient, loadingPatient, latestWorkflow } = usePatient();
  const router = useRouter();

  if (loadingPatient || !patient) {
    return <div className="text-sm text-neuro-muted mb-4">Loading patient…</div>;
  }

  const bucket = latestWorkflow?.agent_results?.risk_stratification_agent?.raw_payload?.overall_bucket;

  return (
    <div className="flex items-start justify-between mb-4 flex-wrap gap-3">
      <div>
        <button onClick={() => router.push("/")} className="text-xs text-neuro-muted mb-1">
          &larr; all patients
        </button>
        <h1 className="text-xl font-semibold">{patient.full_name}</h1>
        <div className="text-sm text-neuro-muted">
          {patient.age ? `${patient.age} yo` : ""} {patient.sex || ""}
          {(patient.conditions || []).map((c: string) => (
            <span key={c} className="badge badge-info ml-1">
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-3">
        {latestWorkflow?.is_emergency && <span className="badge badge-critical">⚠ Emergency flagged</span>}
        {bucket && <RiskBadge bucket={bucket} size="lg" />}
      </div>
    </div>
  );
}

export default function PatientLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const params = useParams();
  const patientId = params.id as string;

  useEffect(() => {
    const { token } = loadSession();
    if (!token) router.push("/login");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <PatientProvider patientId={patientId}>
      <PatientHeader />
      <TabNav patientId={patientId} />
      {children}
    </PatientProvider>
  );
}
