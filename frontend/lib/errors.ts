// Turns the raw "<status> <statusText>: <body>" error thrown by lib/api.ts into a message a person can act on.
// Pure and framework-free so it can be unit-tested. Never echoes server internals beyond a short validation hint.

export function parseApiError(raw: string): { status: number | null; detail: string | null } {
  const m = /^(\d{3})\b[^:]*:\s*([\s\S]*)$/.exec(raw || "");
  if (!m) return { status: null, detail: null };
  const status = Number(m[1]);
  let detail: string | null = m[2] || null;
  try {
    const j = JSON.parse(m[2]);
    if (typeof j?.detail === "string") detail = j.detail;
    else if (Array.isArray(j?.detail) && j.detail.length) {
      const first = j.detail[0];
      const field = Array.isArray(first?.loc) ? String(first.loc[first.loc.length - 1]) : "";
      detail = `${field ? field + ": " : ""}${first?.msg ?? "invalid value"}`;
    }
  } catch {
    /* body was not JSON; keep the text */
  }
  return { status, detail };
}

export function friendlyMessage(raw: string): string {
  const { status, detail } = parseApiError(raw);
  if (status === null) return "Could not reach the server. Check your connection and try again.";
  switch (status) {
    case 401:
      return "Email or password is incorrect.";
    case 403:
      return detail && /invite/i.test(detail) ? detail : "You do not have permission to do that.";
    case 404:
      return "That record was not found.";
    case 422:
      return detail ? `Please check your input — ${detail}` : "Please check your input and try again.";
    case 429:
      return "Too many attempts. Please wait a minute before trying again.";
    default:
      return status >= 500 ? "The server had a problem. Please try again shortly." : (detail || "Something went wrong.");
  }
}
