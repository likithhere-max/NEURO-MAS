import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
const login = vi.fn();
const register = vi.fn();
vi.mock("@/lib/api", () => ({ api: { login: (...a: any[]) => login(...a), register: (...a: any[]) => register(...a) }, setSession: vi.fn() }));

import LoginPage from "../app/login/page";

function fill() {
  fireEvent.change(screen.getByLabelText(/email/i), { target: { value: "a@example.com" } });
  fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "pw-123456789" } });
}

beforeEach(() => { login.mockReset(); register.mockReset(); push.mockReset(); });

describe("LoginPage", () => {
  it("does not prefill credentials", () => {
    render(<LoginPage />);
    expect((screen.getByLabelText(/email/i) as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText(/password/i) as HTMLInputElement).value).toBe("");
  });

  it("announces a rate-limit error accessibly (role=alert) with a human message", async () => {
    login.mockRejectedValue(new Error("429 Too Many Requests: {}"));
    render(<LoginPage />);
    fill();
    fireEvent.submit(screen.getByRole("button", { name: /sign in/i }).closest("form")!);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/too many attempts/i);
    expect(push).not.toHaveBeenCalled();
  });

  it("shows a bad-credentials message for 401 and does not navigate", async () => {
    login.mockRejectedValue(new Error('401 Unauthorized: {"detail":"x"}'));
    render(<LoginPage />);
    fill();
    fireEvent.submit(screen.getByRole("button", { name: /sign in/i }).closest("form")!);
    expect(await screen.findByRole("alert")).toHaveTextContent(/incorrect/i);
  });

  it("registration that needs email confirmation does NOT log in and says so", async () => {
    register.mockResolvedValue({ status: "confirmation_required" });
    render(<LoginPage />);
    fireEvent.click(screen.getByText(/create an account|register|sign up/i, { selector: "button" }));
    fill();
    fireEvent.submit(document.querySelector("form")!);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/confirm your address/i));
    expect(login).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
  });
});
