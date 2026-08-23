"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, loadSession } from "@/lib/api";
import AgentActivityPanel from "@/components/AgentActivityPanel";
import ConsensusPanel from "@/components/ConsensusPanel";
import TimelinePanel from "@/components/TimelinePanel";
import PersonalizationPanel from "@/components/PersonalizationPanel";
import ActionsPanel from "@/components/ActionsPanel";
import AlertCenter from "@/components/AlertCenter";

const PRESETS: Record<string, any> = {
  epilepsy_emergency: {
    label: "Preset: Epilepsy emergency event",
    trigger_reason: "patient_reported_unusual_event",
    observations: {
      symptoms: [{ name: "seizure", severity: "severe", free_text: "seizure not stopping, status epilepticus, over 5 minutes and not regaining consciousness" }],
      medication_logs: [{ name: "levetiracetam", taken: false, scheduled_time: "08:00" }],
      wearable: {},
      vitals: {},
      caregiver_notes: "Caregiver reports the seizure has continued for several minutes without stopping.",
    },
  },
  parkinsons_deterioration: {
    label: "Preset: Parkinson's tremor/gait deterioration",
    trigger_reason: "wearable_trend_deviation",
    observations: {
      symptoms: [{ name: "tremor", severity: "moderate", free_text: "increased hand tremor and slower walking today" }],
      medication_logs: [{ name: "levodopa", taken: false, scheduled_time: "07:00" }],
      wearable: { tremor_amplitude: 1.7, gait_speed_m_s: 0.85 },
      vitals: {},
      caregiver_notes: null,
    },
  },
  routine_checkin: {
    label: "Preset: Routine low-risk check-in",
    trigger_reason: "routine_check_in",
    observations: {
      symptoms: [{ name: "mild headache", severity: "low", free_text: "slight headache, otherwise feeling okay" }],
      medication_logs: [],
      wearable: {},
      vitals: {},
      caregiver_notes: null,
    },
  },
};

export default function PatientDashboard() {
  const router = useRouter();
  const params = useParams();
  const patientId = params.id as string;

  const [role, setRole] = useState<string | null>(null);
  const [patient, setPatient] = useState<any>(null);
  const [workflow, setWorkflow] = useState<any>(null);
  const [timeline, setTimeline] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preset, setPreset] = useState("epilepsy_emergency");

  async function refreshSideData() {
    const [t, a] = await Promise.all([api.getTimeline(patientId), api.getPatientAlerts(patientId)]);
    setTimeline(t);
    setAlerts(a);
  }

  useEffect(() => {
    const { token, role } = loadSession();
    if (!token) {
      router.push("/login");
      return;
    }
    setRole(role);
    api.getPatient(patientId).then(setPatient).catch((e) => setError(e.message));
    refreshSideData().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  async function runWorkflow() {
    setRunning(true);
    setError(null);
    try {
      const result = await api.triggerWorkflow(patientId, PRESETS[preset]);
      setWorkflow(result);
      await refreshSideData();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }

  if (!patient) {
    return <div className="text-sm text-neuro-muted">Loading patient...</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <button onClick={() => router.push("/")} className="text-xs text-neuro-muted mb-1">&larr; all patients</button>
          <h1 className="text-xl font-semibold">{patient.full_name}</h1>
          <div className="text-sm text-neuro-muted">
            {patient.age ? `${patient.age} yo` : ""} {patient.sex || ""} ·{" "}
            {patient.conditions.map((c: string) => <span key={c} className="badge badge-info ml-1">{c}</span>)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select value={preset} onChange={(e) => setPreset(e.target.value)}
            className="bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm">
            {Object.entries(PRESETS).map(([key, p]: any) => (
              <option key={key} value={key}>{p.label}</option>
            ))}
          </select>
          <button onClick={runWorkflow} disabled={running}
            className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50">
            {running ? "Agents working..." : "Trigger assessment"}
          </button>
        </div>
      </div>

      {error && <div className="text-neuro-critical text-sm mb-3">{error}</div>}

      {workflow?.is_emergency && (
        <div className="mb-4 rounded-lg border border-neuro-critical bg-neuro-critical/10 px-4 py-3 text-sm text-neuro-critical font-medium">
          ⚠ Deterministic emergency detector triggered — Level 4 escalation fired automatically, independent of
          agent consensus.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <AgentActivityPanel agentResults={workflow?.agent_results || {}} />
          <ConsensusPanel consensus={workflow?.consensus} />
          <PersonalizationPanel agentResults={workflow?.agent_results || {}} />
        </div>
        <div className="space-y-6">
          <ActionsPanel actions={workflow?.proposed_actions || []} role={role} onChanged={refreshSideData} />
          <AlertCenter alerts={alerts} onChanged={refreshSideData} />
          <TimelinePanel events={workflow?.audit_trail || timeline} />
        </div>
      </div>
    </div>
  );
}
