// Shown instead of a prediction when the model abstained (input outside the validated domain, or invalid).
// No label and no probability are displayed in this state.
export interface DomainReason { code: string; check?: string; feature?: string | null; detail?: string }

export default function AbstainNotice({ domain }: { domain: { status?: string; reasons?: DomainReason[] } | null | undefined }) {
  const reasons = domain?.reasons ?? [];
  const invalid = domain?.status === "abstain_invalid_input";
  return (
    <div role="status" className="border border-neuro-warn rounded-lg px-3 py-2 text-sm">
      <div className="font-semibold text-neuro-warn">
        No prediction made — {invalid ? "the input is not valid for this model" : "the input is outside the range this model was trained on"}
      </div>
      <p className="text-xs text-neuro-muted mt-1">
        Showing a number here would be misleading. This is the model declining to answer, not a normal or healthy result.
      </p>
      {reasons.length > 0 && (
        <ul className="text-xs text-neuro-muted mt-2 list-disc pl-4">
          {reasons.slice(0, 5).map((r, i) => (
            <li key={`${r.code}-${r.feature ?? ""}-${i}`}>{r.feature ? `${r.feature}: ` : ""}{r.detail ?? r.code}</li>
          ))}
          {reasons.length > 5 && <li>…and {reasons.length - 5} more</li>}
        </ul>
      )}
    </div>
  );
}
