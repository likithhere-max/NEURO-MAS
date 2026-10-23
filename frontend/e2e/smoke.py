"""
UI smoke test (Playwright for Python, Chromium) against a production build served by `next start`.

SCOPE, stated plainly: the backend API is MOCKED with route interception. This verifies frontend behaviour
(login flow, accessible error messages, abstention rendering, no crash on null predictions). It does NOT verify the
backend, Supabase, or any live integration.

Usage:  npm run build && npx next start -p 3100 &   then   python3 e2e/smoke.py http://localhost:3100
Exit code 0 only if every check passes.
"""
import json
import sys

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3100"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def fulfil(route, status, body):
    route.fulfill(status=status, content_type="application/json", body=json.dumps(body),
                  headers={"access-control-allow-origin": "*", "access-control-allow-headers": "*",
                           "access-control-allow-methods": "*"})


def handler_factory(state):
    def handler(route):
        req = route.request
        url, method = req.url, req.method
        if method == "OPTIONS":
            return fulfil(route, 204, {})
        if url.endswith("/auth/login"):
            state["logins"] += 1
            if state["logins"] == 1:
                return fulfil(route, 429, {"detail": "too many"})
            return fulfil(route, 200, {"access_token": "test-token", "role": "clinician", "token_type": "bearer"})
        if url.endswith("/patients"):
            return fulfil(route, 200, [])
        if url.endswith("/diseases") or "/diseases" in url:
            return fulfil(route, 200, {"diseases": []})
        if url.endswith("/info"):
            return fulfil(route, 200, {"model": "stub", "metrics_status": "NOT_INDEPENDENTLY_VALIDATED", "limitations": []})
        if method == "POST" and (url.endswith("/demo/parkinsons-classifier") or url.endswith("/demo/parkinsons-risk-classifier")):
            return fulfil(route, 200, {
                "predicted_class": None, "predicted_label": None, "probability_parkinsons": None,
                "probability_healthy": None, "probability_worsening": None, "probability_stable_or_improving": None,
                "abstained": True, "disclaimer": "demo",
                "domain": {"status": "abstain_out_of_domain", "in_domain": False, "reasons": [
                    {"code": "abstain_out_of_domain", "check": "range", "feature": "HNR", "detail": "99 outside training range"}]}})
        return fulfil(route, 200, {})
    return handler


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    state = {"logins": 0}
    page.route("**/api/v1/**", handler_factory(state))

    page.goto(BASE + "/login", wait_until="networkidle")
    page.wait_for_timeout(500)  # let React hydrate before interacting
    check("login: fields are not prefilled",
          page.get_by_label("Email").input_value() == "" and page.get_by_label("Password").input_value() == "")
    page.get_by_label("Email").fill("clin@example.com")
    page.get_by_label("Password").fill("not-a-real-password")
    page.get_by_role("button", name="Sign in").click()
    # Next.js adds its own empty role=alert route announcer, so select our message by content.
    alert = page.get_by_role("alert").filter(has_text="attempts")
    alert.wait_for(timeout=5000)
    check("login: 429 shows an accessible, human message", "too many attempts" in alert.inner_text().lower(), alert.inner_text())

    page.get_by_role("button", name="Sign in").click()
    page.wait_for_url(BASE + "/", timeout=8000)
    check("login: successful sign-in navigates home", page.url.rstrip("/") == BASE.rstrip("/"))

    page.goto(BASE + "/ml-lab", wait_until="networkidle")
    page.wait_for_timeout(500)
    page.get_by_role("button", name="Run classifier").first.click()
    status = page.get_by_role("status").filter(has_text="No prediction made").first
    status.wait_for(timeout=5000)
    txt = status.inner_text().lower()
    check("ml-lab: abstained voice result shows 'no prediction made'", "no prediction made" in txt, txt)
    check("ml-lab: abstention shows no 'predicted:' label", page.get_by_text("predicted:", exact=False).count() == 0)
    check("ml-lab: abstention reason is shown", "hnr" in page.content().lower())
    page.get_by_role("button", name="Run classifier").nth(1).click()
    page.wait_for_timeout(800)
    check("ml-lab: risk abstention does not crash the page", not errors, "; ".join(errors))
    check("ml-lab: no 'null' text leaked", page.get_by_text("null", exact=False).count() == 0)
    browser.close()

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
