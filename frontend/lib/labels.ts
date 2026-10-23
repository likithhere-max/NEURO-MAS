// Pure presentation helpers (unit-tested in tests/labels.test.ts). They exist so the UI never states more than the
// backend established: delivery wording follows the outbox state machine, and consensus scores are shown as an
// uncalibrated support index, never as a percentage probability.

export const ACTION_STATUS_LABEL: Record<string, string> = {
  proposed: "proposed · not sent",
  pending_approval: "awaiting clinician approval",
  approved: "approved · not yet sent",
  queued: "queued for delivery · not yet sent",
  sent_unconfirmed: "handed to provider · delivery not confirmed",
  delivered: "delivered",
  executed: "executed",
  acknowledged: "acknowledged by recipient",
  delivery_failed: "delivery FAILED · needs attention",
  failed: "delivery FAILED · needs attention",
  escalated: "escalated",
  rejected: "rejected",
};

export function actionStatusLabel(status: string | null | undefined): string {
  if (!status) return "unknown status";
  return ACTION_STATUS_LABEL[status] ?? status.replaceAll("_", " ");
}

export type Tone = "ok" | "warn" | "critical" | "info";

/** Failed or unconfirmed deliveries must never look like success. */
export function actionStatusTone(status: string | null | undefined): Tone {
  switch (status) {
    case "executed":
    case "delivered":
    case "acknowledged":
      return "ok";
    case "delivery_failed":
    case "failed":
      return "critical";
    case "sent_unconfirmed":
    case "queued":
    case "proposed":
    case "pending_approval":
    case "approved":
    case "escalated":
      return "warn";
    default:
      return "info";
  }
}

export interface HypothesisLike {
  hypothesis: string;
  consensus_score: number;
  status?: string;
  contradiction_index?: number;
}

/** Support index in [0,1] -> display string. When the backend says it is not a calibrated probability
 *  (the only case currently produced) we show a plain index, with no % sign. */
export function formatSupport(h: HypothesisLike, calibrated: boolean | undefined): string {
  const v = Number.isFinite(h.consensus_score) ? Math.min(1, Math.max(0, h.consensus_score)) : 0;
  return calibrated ? `${Math.round(v * 100)}%` : `${v.toFixed(2)} index`;
}

export function hypothesisStatusLabel(status: string | undefined): string {
  switch (status) {
    case "supported": return "supported";
    case "contradicted": return "contradicted by independent evidence";
    case "unsupported": return "no supporting evidence";
    default: return "";
  }
}
