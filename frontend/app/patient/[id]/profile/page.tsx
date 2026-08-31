"use client";

import { usePatient } from "@/components/PatientContext";

export default function ProfilePage() {
  const { patient } = usePatient();
  if (!patient) return <div className="text-sm text-neuro-muted">Loading…</div>;

  return (
    <div className="max-w-2xl space-y-4">
      <div className="card">
        <h1 className="font-semibold text-lg mb-3">Patient profile</h1>
        <dl className="text-sm grid grid-cols-2 gap-y-2">
          <dt className="text-neuro-muted">Full name</dt>
          <dd>{patient.full_name}</dd>
          <dt className="text-neuro-muted">Age</dt>
          <dd>{patient.age ?? "—"}</dd>
          <dt className="text-neuro-muted">Sex</dt>
          <dd>{patient.sex ?? "—"}</dd>
          <dt className="text-neuro-muted">Known conditions</dt>
          <dd>{patient.conditions?.length ? patient.conditions.join(", ") : "none recorded"}</dd>
        </dl>
      </div>
      <div className="text-[11px] text-neuro-muted">
        Editing profile fields and adding medications/allergies is available via the backend API
        (POST /patients/&#123;id&#125;/medications, /allergies) but not yet wired into this UI.
      </div>
    </div>
  );
}
