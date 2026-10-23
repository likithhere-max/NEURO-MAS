"use client";

// Shared "patient state" for the frontend (2-day review sprint rebuild).
//
// This is the UI-side analogue of the backend's shared blackboard
// (app/schemas/patient_state.py PatientState): every tab under
// /patient/[id]/* reads from and writes through THIS one context instead of
// independently re-fetching and re-deriving the same data, so "symptoms
// entered -> risk changes -> chat knows -> dashboard knows" is one real
// data flow, not several disconnected screens hitting the API separately.

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api } from "@/lib/api";

type Ctx = {
  patientId: string;
  patient: any;
  loadingPatient: boolean;
  assessments: any[];
  latestAssessment: any | null;
  latestWorkflow: any | null;
  loadingWorkflow: boolean;
  alerts: any[];
  timeline: any[];
  chatHistory: any[];
  chatLoading: boolean;
  error: string | null;
  runAssessment: (payload: any) => Promise<any>;
  sendChatMessage: (message: string) => Promise<void>;
  refreshAll: () => Promise<void>;
};

const PatientCtx = createContext<Ctx | null>(null);

// /workflows/{id} (GET) returns agent_events as a LIST; the trigger
// endpoints (/workflows/neurological-assessment/{id},
// /workflows/symptom-assessment/{id}) return agent_results as a DICT keyed
// by agent name. Every tab component expects the dict shape (that's what
// the pre-existing AgentActivityPanel/ConsensusPanel/etc. already consume),
// so this normalizes the GET shape to match rather than teaching every
// consumer both shapes.
function normalizeWorkflowFromGet(wf: any) {
  const agent_results: Record<string, any> = {};
  for (const e of wf.agent_events || []) {
    agent_results[e.agent_name] = { ...e.result, agent_name: e.agent_name, status: e.status, confidence: e.confidence };
  }
  return {
    workflow_id: wf.workflow_id,
    patient_id: wf.patient_id,
    status: wf.status,
    is_emergency: wf.is_emergency,
    trigger_reason: wf.trigger_reason,
    started_at: wf.started_at,
    completed_at: wf.completed_at,
    consensus: wf.consensus,
    degraded: wf.degraded,
    failed_agents: wf.failed_agents,
    agent_results,
    proposed_actions: wf.actions || [],
    audit_trail: wf.audit_trail || [],
  };
}

export function PatientProvider({ patientId, children }: { patientId: string; children: React.ReactNode }) {
  const [patient, setPatient] = useState<any>(null);
  const [loadingPatient, setLoadingPatient] = useState(true);
  const [assessments, setAssessments] = useState<any[]>([]);
  const [latestAssessment, setLatestAssessment] = useState<any | null>(null);
  const [latestWorkflow, setLatestWorkflow] = useState<any | null>(null);
  const [loadingWorkflow, setLoadingWorkflow] = useState(true);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [timeline, setTimeline] = useState<any[]>([]);
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCore = useCallback(async () => {
    setLoadingPatient(true);
    try {
      setPatient(await api.getPatient(patientId));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingPatient(false);
    }
  }, [patientId]);

  const loadAssessments = useCallback(async () => {
    setLoadingWorkflow(true);
    try {
      const history = await api.getSymptomAssessmentHistory(patientId);
      setAssessments(history || []);
      const newest = (history || [])[0] || null;
      setLatestAssessment(newest);
      if (newest?.workflow_id) {
        const wf = await api.getWorkflow(newest.workflow_id);
        setLatestWorkflow(normalizeWorkflowFromGet(wf));
      } else {
        setLatestWorkflow(null);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoadingWorkflow(false);
    }
  }, [patientId]);

  const loadSideData = useCallback(async () => {
    try {
      const [t, a] = await Promise.all([api.getTimeline(patientId), api.getPatientAlerts(patientId)]);
      setTimeline(t);
      setAlerts(a);
    } catch {
      // non-fatal: timeline/alerts are supplementary, not required for the
      // main tabs to function
    }
  }, [patientId]);

  const loadChat = useCallback(async () => {
    try {
      setChatHistory((await api.getChatHistory(patientId)) || []);
    } catch {
      // non-fatal
    }
  }, [patientId]);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadCore(), loadAssessments(), loadSideData(), loadChat()]);
  }, [loadCore, loadAssessments, loadSideData, loadChat]);

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  async function runAssessment(payload: any) {
    const result = await api.triggerSymptomAssessment(patientId, payload);
    const assessmentRecord = result.assessment || {
      id: undefined,
      patient_id: patientId,
      workflow_id: result.workflow?.workflow_id,
      free_text: payload.free_text,
      extracted_json: result.extracted,
      created_at: new Date().toISOString(),
    };
    setLatestAssessment(assessmentRecord);
    setLatestWorkflow(result.workflow);
    setAssessments((prev) => [assessmentRecord, ...prev]);
    await loadSideData();
    return result;
  }

  async function sendChatMessage(message: string) {
    setChatLoading(true);
    setChatHistory((prev) => [...prev, { role: "user", content: message, created_at: new Date().toISOString() }]);
    try {
      const result = await api.sendChat(patientId, message);
      setChatHistory((prev) => [...prev, { role: "assistant", content: result.answer, created_at: new Date().toISOString(), is_emergency: !!result.is_emergency } as any]);
    } catch (e: any) {
      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: `Something went wrong reaching the assistant: ${e.message}`, created_at: new Date().toISOString() },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <PatientCtx.Provider
      value={{
        patientId,
        patient,
        loadingPatient,
        assessments,
        latestAssessment,
        latestWorkflow,
        loadingWorkflow,
        alerts,
        timeline,
        chatHistory,
        chatLoading,
        error,
        runAssessment,
        sendChatMessage,
        refreshAll,
      }}
    >
      {children}
    </PatientCtx.Provider>
  );
}

export function usePatient() {
  const ctx = useContext(PatientCtx);
  if (!ctx) throw new Error("usePatient must be used within a PatientProvider (app/patient/[id]/layout.tsx)");
  return ctx;
}
