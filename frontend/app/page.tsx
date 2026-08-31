"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, loadSession } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [patients, setPatients] = useState<any[]>([]);
  const [role, setRole] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newAge, setNewAge] = useState("60");
  const [newConditionId, setNewConditionId] = useState("");
  const [diseases, setDiseases] = useState<any[]>([]);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [p, d] = await Promise.all([api.listPatients(), api.listDiseases()]);
      setPatients(p);
      setDiseases(d.diseases || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    const { token, role } = loadSession();
    if (!token) {
      router.push("/login");
      return;
    }
    setRole(role);
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function createPatient(e: React.FormEvent) {
    e.preventDefault();
    const created = await api.createPatient({ full_name: newName, age: Number(newAge) });
    // create_patient() deliberately never persists a condition string itself
    // (ambiguous: disease_id vs. preferred_name -- see the backend's own
    // comment). If the clinician picked a known pre-existing condition here,
    // record it through the unambiguous /conditions endpoint instead, so
    // this dropdown actually does something rather than being silently
    // discarded (that used to happen -- fixed in the 2-day review sprint).
    if (newConditionId) {
      const disease = diseases.find((d) => d.disease_id === newConditionId);
      if (disease) {
        await api.addCondition(created.id, {
          disease_id: disease.disease_id,
          preferred_name: disease.preferred_name,
        });
      }
    }
    setNewName("");
    setNewConditionId("");
    refresh();
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2">
        <h1 className="text-xl font-semibold mb-1">Patients</h1>
        <p className="text-sm text-neuro-muted mb-4">
          A Multi-Agent Collaborative System for Autonomous and Personalized Neurological Healthcare.
          Select a patient to run a neurological assessment workflow and watch the agents collaborate.
        </p>
        {error && <div className="text-neuro-critical text-sm mb-3">{error}</div>}
        {loading ? (
          <div className="text-sm text-neuro-muted">Loading...</div>
        ) : patients.length === 0 ? (
          <div className="card text-sm text-neuro-muted">
            No patients yet. Create one on the right, or run{" "}
            <code className="text-neuro-accent2">python -m app.seed.demo_scenarios</code> on the backend for
            ready-made Epilepsy / Parkinson&apos;s demo patients.
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 gap-3">
            {patients.map((p) => (
              <button
                key={p.id}
                onClick={() => router.push(`/patient/${p.id}`)}
                className="card text-left hover:border-neuro-accent transition-colors"
              >
                <div className="font-medium">{p.full_name}</div>
                <div className="text-xs text-neuro-muted">{p.age ? `${p.age} yo` : ""} {p.sex || ""}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {p.conditions.map((c: string) => (
                    <span key={c} className="badge badge-info">{c}</span>
                  ))}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="card">
          <h2 className="font-semibold mb-3">New patient</h2>
          <form onSubmit={createPatient} className="space-y-3">
            <input required className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
              placeholder="Full name" value={newName} onChange={(e) => setNewName(e.target.value)} />
            <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
              placeholder="Age" type="number" value={newAge} onChange={(e) => setNewAge(e.target.value)} />
            <div>
              <label className="text-xs text-neuro-muted block mb-1">
                Known pre-existing condition (optional — clinician-documented history, not a diagnosis)
              </label>
              <select className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                value={newConditionId} onChange={(e) => setNewConditionId(e.target.value)}>
                <option value="">None recorded yet</option>
                {diseases.map((d: any) => (
                  <option key={d.disease_id} value={d.disease_id}>{d.preferred_name}</option>
                ))}
              </select>
            </div>
            <button className="w-full bg-neuro-accent text-neuro-bg font-medium rounded-lg py-2 text-sm">
              Create patient
            </button>
          </form>
          <div className="text-[11px] text-neuro-muted mt-3">
            To assess new symptoms, create the patient here, then use Symptom Assessment on their dashboard —
            diagnosis is never made by picking from this list.
          </div>
        </div>
        <div className="text-[11px] text-neuro-muted mt-3">Signed in as role: {role}</div>
      </div>
    </div>
  );
}
