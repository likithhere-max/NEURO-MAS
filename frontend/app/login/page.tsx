"use client";

import { useState } from "react";
import { friendlyMessage } from "../../lib/errors";
import { useRouter } from "next/navigation";
import { api, setSession } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState("clinician");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      if (mode === "register") {
        const created = await api.register(email, password, fullName, role, inviteCode);
        if (created && created.status === "confirmation_required") {
          // Email verification is enforced: no session exists until the address is confirmed.
          setNotice("Account created. Check your email and confirm your address, then sign in.");
          setMode("login");
          return;
        }
      }
      const data = await api.login(email, password);
      setSession(data.access_token, data.role);
      router.push("/");
    } catch (err: any) {
      setError(friendlyMessage(String(err?.message ?? "")));
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
                placeholder="Full name" aria-label="Full name" autoComplete="name" value={fullName} onChange={(e) => setFullName(e.target.value)} />
              <select className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                aria-label="Role" value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="clinician">Clinician</option>
                <option value="patient">Patient</option>
                <option value="caregiver">Caregiver</option>
              </select>
              {role === "clinician" && (
                <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                  placeholder="Clinician invite code (required on deployed servers)" aria-label="Clinician invite code" value={inviteCode}
                  onChange={(e) => setInviteCode(e.target.value)} />
              )}
            </>
          )}
          <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            placeholder="Email" aria-label="Email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <input className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
            placeholder="Password" aria-label="Password" type="password" required minLength={mode === "register" ? 8 : undefined} autoComplete={mode === "register" ? "new-password" : "current-password"} value={password} onChange={(e) => setPassword(e.target.value)} />
          {error && <div role="alert" className="text-neuro-critical text-xs">{error}</div>}
          {notice && <div role="status" className="text-neuro-accent text-xs">{notice}</div>}
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
