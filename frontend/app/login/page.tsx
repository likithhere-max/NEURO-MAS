"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, setSession } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("clinician@neuromas.demo");
  const [password, setPassword] = useState("demo12345");
  const [fullName, setFullName] = useState("Dr. Demo Clinician");
  const [role, setRole] = useState("clinician");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      if (mode === "register") {
        await api.register(email, password, fullName, role, inviteCode);
      }
      const data = await api.login(email, password);
      setSession(data.access_token, data.role);
      router.push("/");
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-md mx-auto mt-16">
      <div className="card">
        <h1 className="text-lg font-semibold mb-1">
          {mode === "login" ? "Sign in" : "Create an account"}
        </h1>
        <p className="text-sm text-neuro-muted mb-4">
          NEURO-MAS is built for clinicians first — patients and caregivers get a
          view of their own data, but every AI-generated finding is reviewed
          against clinician oversight.
        </p>
        <form onSubmit={handleSubmit} className="space-y-3">
          {mode === "register" && (
            <>
              <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                placeholder="Full name" value={fullName} onChange={(e) => setFullName(e.target.value)} />
              <select className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="clinician">Clinician</option>
                <option value="patient">Patient</option>
                <option value="caregiver">Caregiver</option>
              </select>
              {role === "clinician" && (
                <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                  placeholder="Clinician invite code (required on deployed servers)" value={inviteCode}
                  onChange={(e) => setInviteCode(e.target.value)} />
              )}
            </>
          )}
          <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            placeholder="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error && <div className="text-neuro-critical text-xs">{error}</div>}
          <button disabled={loading} className="w-full bg-neuro-accent text-neuro-bg font-medium rounded-lg py-2 text-sm disabled:opacity-50">
            {loading ? "Please wait..." : mode === "login" ? "Sign in" : "Register & sign in"}
          </button>
        </form>
        <button
          className="text-xs text-neuro-muted mt-3 underline"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}
