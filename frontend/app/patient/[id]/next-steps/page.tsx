"use client";

import Link from "next/link";
import { usePatient } from "@/components/PatientContext";

function Section({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="card">
      <h2 className="font-semibold mb-2">
        {n}. {title}
      </h2>
      <div className="text-sm text-neuro-text">{children}</div>
    </div>
  );
}

export default function NextStepsPage() {
  const { patientId, latestWorkflow } = usePatient();

  if (!latestWorkflow) {
    return (
      <div className="card text-center py-12">
        <div className="text-lg font-semibold mb-2">No assessment yet</div>
        <Link
          href={`/patient/${patientId}/assessment`}
          className="inline-block bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm"
        >
          Start Symptom Assessment
        </Link>
      </div>
    );
  }

  const risk = latestWorkflow.agent_results?.risk_stratification_agent?.raw_payload || {};
  const diagnostic = latestWorkflow.agent_results?.neuro_diagnostic_agent?.raw_payload || {};
  const personalization = latestWorkflow.agent_results?.personalization_agent?.raw_payload || {};
  const topCandidate = diagnostic.candidate_conditions?.[0];
  const bucket = risk.overall_bucket;

  return (
    <div className="max-w-3xl space-y-6">
      {latestWorkflow.is_emergency && (
        <div className="rounded-lg border border-neuro-critical bg-neuro-critical/10 px-4 py-4 text-sm text-neuro-critical font-medium">
          ⚠ This assessment was flagged by the deterministic emergency detector. Seek immediate emergency medical
          care now rather than relying on this guidance.
        </div>
      )}

      <Section n={1} title="Immediate attention">
        {latestWorkflow.is_emergency || bucket === "critical"
          ? "Current findings indicate this warrants urgent evaluation — seek emergency or urgent care."
          : "Nothing in the current findings indicates an immediate emergency, based on the deterministic safety rules."}
      </Section>

      <Section n={2} title="Medical follow-up">
        {topCandidate
          ? "A neurologist (or your primary care physician for an initial referral) would be an appropriate next point of contact, given the candidate condition(s) under consideration."
          : "A primary care physician is a reasonable starting point for further evaluation."}
      </Section>

      <Section n={3} title="Possible evaluation">
        <ul className="list-disc list-inside text-sm space-y-1">
          <li>Clinical history and physical/neurological examination</li>
          <li>Discussion of symptom timeline, triggers, and progression with a clinician</li>
          <li>Condition-specific workup as determined by your clinician (see each candidate&apos;s Disease Info page for typical evaluation methods)</li>
        </ul>
      </Section>

      <Section n={4} title="Information to prepare">
        <ul className="list-disc list-inside text-sm space-y-1">
          <li>When symptoms started and how they&apos;ve changed</li>
          <li>Severity, frequency, and specific triggers or relieving factors</li>
          <li>Any current medications and known allergies</li>
          <li>Relevant family or personal medical history</li>
        </ul>
      </Section>

      <Section n={5} title="Questions to ask a doctor">
        <ul className="list-disc list-inside text-sm space-y-1">
          <li>What are the possible explanations for these symptoms?</li>
          <li>What tests or evaluations would help narrow this down?</li>
          <li>What symptoms should prompt me to seek care sooner?</li>
          <li>What can I do in the meantime?</li>
        </ul>
      </Section>

      <Section n={6} title="Warning signs — seek urgent care if these occur">
        <ul className="list-disc list-inside text-sm space-y-1 text-neuro-critical">
          {(diagnostic.red_flags || []).length > 0 ? (
            diagnostic.red_flags.map((r: string) => <li key={r}>{r}</li>)
          ) : (
            <li>Sudden weakness, difficulty speaking, loss of consciousness, or seizures that don&apos;t stop</li>
          )}
        </ul>
      </Section>

      {personalization.monitoring_plan?.length > 0 && (
        <div className="card">
          <h2 className="font-semibold mb-2">Personalized monitoring plan</h2>
          <ul className="list-disc list-inside text-sm space-y-1">
            {personalization.monitoring_plan.map((p: string, i: number) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="text-[11px] text-neuro-muted border-t border-neuro-border pt-3">
        NEURO-MAS is a decision-support research prototype, not an autonomous doctor. This guidance does not replace
        professional medical evaluation, diagnosis, or treatment.
      </div>
    </div>
  );
}
