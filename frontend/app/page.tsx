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
  const [newCondition, setNewCondition] = useState("Epilepsy (focal and generalized)");
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
    await api.createPatient({ full_name: newName, age: Number(newAge), conditions: [newCondition] });
    setNewName("");
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
            <select className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
              value={newCondition} onChange={(e) => setNewCondition(e.target.value)}>
              {diseases.map((d: any) => (
                <option key={d.disease_id} value={d.preferred_name}>{d.preferred_name}</option>
              ))}
            </select>
            <button className="w-full bg-neuro-accent text-neuro-bg font-medium rounded-lg py-2 text-sm">
              Create patient
            </button>
          </form>
        </div>
        <div className="text-[11px] text-neuro-muted mt-3">Signed in as role: {role}</div>
      </div>
    </div>
  );
}
