// Thin fetch wrapper for the NEURO-MAS FastAPI backend. Auth token lives in
// memory + localStorage is deliberately NOT used here (Claude artifact/browser
// storage rules aside, a real clinical frontend should use httpOnly cookies or
// a proper session -- localStorage token storage is a prototype convenience,
// flagged in docs/SIMULATED_VS_REAL.md).

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

let inMemoryToken: string | null = null;
let inMemoryRole: string | null = null;

export function setSession(token: string, role: string) {
  inMemoryToken = token;
  inMemoryRole = role;
  if (typeof window !== "undefined") {
    sessionStorage.setItem("neuromas_token", token);
    sessionStorage.setItem("neuromas_role", role);
  }
}

export function loadSession() {
  if (typeof window !== "undefined") {
    inMemoryToken = sessionStorage.getItem("neuromas_token");
    inMemoryRole = sessionStorage.getItem("neuromas_role");
  }
  return { token: inMemoryToken, role: inMemoryRole };
}

export function clearSession() {
  inMemoryToken = null;
  inMemoryRole = null;
  if (typeof window !== "undefined") {
    sessionStorage.removeItem("neuromas_token");
    sessionStorage.removeItem("neuromas_role");
  }
}

async function request(path: string, options: RequestInit = {}) {
  const { token } = loadSession();
  const headers: Record<string, string> = {
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers as Record<string, string> | undefined),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

export const api = {
  register: (email: string, password: string, full_name: string, role: string) =>
    request("/auth/register", { method: "POST", body: JSON.stringify({ email, password, full_name, role }) }),

  login: async (email: string, password: string) => {
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  listPatients: () => request("/patients"),
  getPatient: (id: string) => request(`/patients/${id}`),
  createPatient: (payload: any) => request("/patients", { method: "POST", body: JSON.stringify(payload) }),
  addObservations: (id: string, payload: any) => request(`/patients/${id}/observations`, { method: "POST", body: JSON.stringify(payload) }),
  addMedication: (id: string, payload: any) => request(`/patients/${id}/medications`, { method: "POST", body: JSON.stringify(payload) }),
  getTimeline: (id: string) => request(`/patients/${id}/timeline`),
  getRisk: (id: string) => request(`/patients/${id}/risk`),
  getPatientAlerts: (id: string) => request(`/patients/${id}/alerts`),
  getExplanations: (id: string, workflowId: string) => request(`/patients/${id}/explanations?workflow_id=${workflowId}`),

  triggerWorkflow: (patientId: string, payload: any) =>
    request(`/workflows/neurological-assessment/${patientId}`, { method: "POST", body: JSON.stringify(payload) }),
  getWorkflow: (id: string) => request(`/workflows/${id}`),

  searchKnowledge: (query: string) => request("/knowledge/search", { method: "POST", body: JSON.stringify({ query, top_k: 5 }) }),
  listDiseases: () => request("/knowledge/diseases"),

  acknowledgeAlert: (id: string) => request(`/alerts/${id}/acknowledge`, { method: "POST", body: JSON.stringify({}) }),
  approveAction: (id: string) => request(`/alerts/actions/${id}/approve`, { method: "POST" }),

  agentsStatus: () => request("/agents/status"),
};
