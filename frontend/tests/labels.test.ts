import { describe, expect, it } from "vitest";
import { actionStatusLabel, actionStatusTone, formatSupport, hypothesisStatusLabel } from "../lib/labels";

describe("action status wording follows the delivery state machine", () => {
  it("never calls an unconfirmed or failed delivery a success", () => {
    for (const s of ["proposed", "queued", "sent_unconfirmed", "approved", "pending_approval"]) {
      expect(actionStatusTone(s)).not.toBe("ok");
      expect(actionStatusLabel(s)).not.toMatch(/^(executed|delivered)$/);
    }
    expect(actionStatusTone("delivery_failed")).toBe("critical");
    expect(actionStatusLabel("delivery_failed")).toMatch(/FAILED/);
  });
  it("only delivered/executed/acknowledged are ok", () => {
    for (const s of ["delivered", "executed", "acknowledged"]) expect(actionStatusTone(s)).toBe("ok");
  });
  it("falls back safely for unknown/missing statuses", () => {
    expect(actionStatusLabel("some_new_state")).toBe("some new state");
    expect(actionStatusLabel(undefined)).toBe("unknown status");
    expect(actionStatusTone("some_new_state")).toBe("info");
  });
});

describe("consensus support formatting", () => {
  const h = { hypothesis: "x", consensus_score: 0.84 };
  it("shows an uncalibrated index with no percent sign", () => {
    expect(formatSupport(h, false)).toBe("0.84 index");
    expect(formatSupport(h, undefined)).not.toContain("%");
  });
  it("only shows a percentage when the backend declares calibration", () => {
    expect(formatSupport(h, true)).toBe("84%");
  });
  it("clamps and survives bad numbers", () => {
    expect(formatSupport({ ...h, consensus_score: 7 }, false)).toBe("1.00 index");
    expect(formatSupport({ ...h, consensus_score: NaN }, false)).toBe("0.00 index");
  });
  it("labels contradicted hypotheses", () => {
    expect(hypothesisStatusLabel("contradicted")).toMatch(/contradicted/);
    expect(hypothesisStatusLabel("legacy")).toBe("");
  });
});

import { agentStatusStyle } from "../lib/labels";
describe("agent status styling", () => {
  it("timeouts and failures are critical, never neutral", () => {
    expect(agentStatusStyle("timed_out").badge).toBe("badge-critical");
    expect(agentStatusStyle("failed").badge).toBe("badge-critical");
  });
  it("abstained / insufficient data are warnings, not success", () => {
    expect(agentStatusStyle("abstained").badge).toBe("badge-warn");
    expect(agentStatusStyle("insufficient_data").badge).toBe("badge-warn");
  });
  it("unknown statuses render as a warning with the raw value, not as ok/info", () => {
    const s = agentStatusStyle("brand_new");
    expect(s.badge).toBe("badge-warn");
    expect(s.label).toBe("brand_new");
  });
});
