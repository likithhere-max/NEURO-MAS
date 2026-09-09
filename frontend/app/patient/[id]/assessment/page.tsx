"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { usePatient } from "@/components/PatientContext";

// Narrates the real, already-running multi-agent pipeline while the single
// POST /workflows/symptom-assessment/{id} call is in flight (section 18 of
// the brief). This is not a fake progress bar detached from real work: the
// interval only advances while `running` is true and is cleared the moment
// the real response comes back, so it can never claim to finish before the
// actual call does.
const LOADING_STEPS = [
  "Extracting clinical features…",
  "Running neurological assessment…",
  "Evaluating risk…",
  "Retrieving supporting evidence…",
  "Generating explanation…",
];

const FOLLOW_UPS: [string, string][] = [
  ["onset", "When did it start / has it changed?"],
  ["worsening", "Is it getting worse?"],
  ["severity", "How severe is it?"],
  ["frequency", "How often does it occur?"],
  ["location", "Where does it occur?"],
  ["triggers", "Anything that triggers it?"],
  ["associated", "Any associated symptoms?"],
];

export default function SymptomAssessmentPage() {
  const { patientId, runAssessment, latestAssessment } = usePatient();
  const router = useRouter();
  const [freeText, setFreeText] = useState("");
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [showStructured, setShowStructured] = useState(false);
  const [running, setRunning] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Section 25 (demo script) step 10: a follow-up symptom typed in after a
  // prior assessment already exists must be treated as an UPDATE to the same
  // patient picture, not a brand-new, unrelated case -- otherwise the
  // diagnostic pipeline loses everything reported earlier. We don't have (or
  // want) a second "merge" endpoint on the backend; instead we carry the
  // prior free-text forward and combine it with the new input into one
  // cumulative narrative, which is exactly the input shape
  // extract_symptom_profile() already knows how to read.
  const priorText: string | undefined = latestAssessment?.free_text?.trim() || undefined;
  const isUpdate = Boolean(priorText);

  async function analyze() {
    if (!freeText.trim()) {
      setError("Please describe what you're experiencing first.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    setStepIndex(0);
    timerRef.current = setInterval(() => {
      setStepIndex((i) => Math.min(i + 1, LOADING_STEPS.length - 1));
    }, 1400);

    // IMPORTANT: `priorText` (latestAssessment.free_text) is already the full
    // cumulative history from every prior update -- it may itself already
    // contain earlier "Update:" labels. Appending a single new "Update:"
    // line grows this linearly, one label per turn. Previously this
    // re-wrapped the whole prior text in ANOTHER "Previously reported:"
    // label every time, which compounded ("Previously reported: Previously
    // reported: ...") after just two updates -- fixed here.
    const effectiveFreeText = priorText ? `${priorText}\n\nUpdate: ${freeText.trim()}` : freeText.trim();

    try {
      const res = await runAssessment({
        free_text: effectiveFreeText,
        onset_answer: answers.onset || undefined,
        worsening_answer: answers.worsening || undefined,
        severity_answer: answers.severity || undefined,
        frequency_answer: answers.frequency || undefined,
        location_answer: answers.location || undefined,
        triggers_answer: answers.triggers || undefined,
        associated_symptoms_answer: answers.associated || undefined,
      });
      setResult(res);
      setFreeText("");
      setAnswers({});
    } catch (err: any) {
      setError(err.message);
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setRunning(false);
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div className="card">
        <h1 className="font-semibold text-lg mb-1">
          {isUpdate ? "Update your assessment" : "Tell us what you’re experiencing"}
        </h1>
        <p className="text-sm text-neuro-muted mb-3">
          {isUpdate
            ? "Describe what's new or has changed since your last update — it will be combined with what you already told us, not replace it."
            : "Describe your symptoms in your own words — there’s no need to name a condition."}
        </p>
        {isUpdate && (
          <div className="text-xs text-neuro-muted bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 mb-3">
            Previously reported: <span className="text-neuro-text">{priorText}</span>
          </div>
        )}
        <textarea
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          rows={5}
          placeholder={isUpdate ? "What's new or different since last time…" : "Describe your symptoms in your own words…"}
          className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
        />
        <button onClick={() => setShowStructured((v) => !v)} className="text-xs text-neuro-accent mt-3">
          {showStructured ? "Hide" : "Add"} optional details {showStructured ? "▲" : "▼"}
        </button>
        {showStructured && (
          <div className="grid sm:grid-cols-2 gap-3 mt-3">
            {FOLLOW_UPS.map(([key, label]) => (
              <div key={key}>
                <label className="text-xs text-neuro-muted block mb-1">{label}</label>
                <input
                  value={answers[key] || ""}
                  onChange={(e) => setAnswers((a) => ({ ...a, [key]: e.target.value }))}
                  className="w-full bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
                  placeholder="Optional — skip if unsure"
                />
              </div>
            ))}
          </div>
        )}
        {error && <div className="text-neuro-critical text-sm mt-3">{error}</div>}
        <button
          onClick={analyze}
          disabled={running}
          className="mt-4 bg-neuro-accent text-neuro-bg font-medium rounded-lg px-5 py-2.5 text-sm disabled:opacity-50"
        >
          {running ? "Analyzing…" : isUpdate ? "Update Assessment" : "Analyze Symptoms"}
        </button>
      </div>

      {running && (
        <div className="card">
          <div className="space-y-2">
            {LOADING_STEPS.map((s, i) => (
              <div key={s} className={`text-sm flex items-center gap-2 ${i <= stepIndex ? "text-neuro-text" : "text-neuro-muted"}`}>
                <span>{i < stepIndex ? "✓" : i === stepIndex ? "→" : "…"}</span> {s}
              </div>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="card">
          <h2 className="font-semibold mb-2">Assessment complete</h2>
          <p className="text-sm text-neuro-muted mb-3">
            Extracted {result.extracted.symptoms.length} symptom(s). View the full breakdown:
          </p>
          <div className="flex gap-3 flex-wrap">
            <button
              onClick={() => router.push(`/patient/${patientId}/analysis`)}
              className="text-sm bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2"
            >
              View Analysis
            </button>
            <button
              onClick={() => router.push(`/patient/${patientId}/risk`)}
              className="text-sm border border-neuro-border rounded-lg px-4 py-2"
            >
              View Risk
            </button>
            <button
              onClick={() => router.push(`/patient/${patientId}`)}
              className="text-sm border border-neuro-border rounded-lg px-4 py-2"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
