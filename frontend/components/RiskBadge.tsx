"use client";

// Maps the backend's internal heuristic bucket names (low/medium/high/
// critical -- see app/agents/risk.py `_bucket()`) to the patient-facing
// labels from the brief (LOW/MODERATE/HIGH/URGENT). The mapping lives in
// exactly one place so every tab that shows a risk badge stays consistent.

const MAP: Record<string, { label: string; badge: string }> = {
  low: { label: "LOW", badge: "badge-ok" },
  medium: { label: "MODERATE", badge: "badge-warn" },
  high: { label: "HIGH", badge: "badge-high" },
  critical: { label: "URGENT", badge: "badge-critical" },
};

export function riskLabel(bucket?: string | null) {
  return MAP[bucket || ""] || { label: "UNKNOWN", badge: "badge-info" };
}

export default function RiskBadge({ bucket, size = "md" }: { bucket?: string | null; size?: "sm" | "md" | "lg" }) {
  const { label, badge } = riskLabel(bucket);
  const sizeClass = size === "lg" ? "text-base px-4 py-2" : size === "sm" ? "text-[10px] px-2 py-0.5" : "text-xs px-3 py-1";
  return <span className={`badge ${badge} ${sizeClass} font-bold tracking-wide`}>{label}</span>;
}
