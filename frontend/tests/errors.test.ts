import { describe, expect, it } from "vitest";
import { friendlyMessage, parseApiError } from "../lib/errors";

describe("parseApiError", () => {
  it("parses status and string detail", () => {
    expect(parseApiError('403 Forbidden: {"detail":"A valid invite code is required"}')).toEqual({
      status: 403, detail: "A valid invite code is required" });
  });
  it("reduces pydantic validation arrays to field + message", () => {
    const raw = '422 Unprocessable Entity: {"detail":[{"loc":["body","email"],"msg":"value is not a valid email"}]}';
    expect(parseApiError(raw).detail).toBe("email: value is not a valid email");
  });
  it("keeps non-JSON bodies as text and tolerates garbage", () => {
    expect(parseApiError("500 Internal Server Error: boom").detail).toBe("boom");
    expect(parseApiError("TypeError: Failed to fetch")).toEqual({ status: null, detail: null });
    expect(parseApiError("")).toEqual({ status: null, detail: null });
  });
});

describe("friendlyMessage", () => {
  it("never leaks server internals for 5xx", () => {
    const m = friendlyMessage('500 Internal Server Error: {"detail":"psycopg2 OperationalError at 10.0.0.5"}');
    expect(m).not.toMatch(/psycopg2|10\.0\.0\.5/);
  });
  it("explains rate limiting", () => expect(friendlyMessage("429 Too Many Requests: x")).toMatch(/wait/i));
  it("treats network failure as unreachable, not as bad credentials", () => {
    expect(friendlyMessage("Failed to fetch")).toMatch(/reach the server/i);
  });
  it("only echoes 403 detail when it is an invite-code hint", () => {
    expect(friendlyMessage('403 Forbidden: {"detail":"invite code required"}')).toMatch(/invite/i);
    expect(friendlyMessage('403 Forbidden: {"detail":"RLS policy xyz"}')).not.toMatch(/RLS/);
  });
});
