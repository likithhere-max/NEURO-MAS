import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import AbstainNotice from "../components/AbstainNotice";

describe("AbstainNotice", () => {
  it("states that no prediction was made and shows reasons, never a label or probability", () => {
    render(<AbstainNotice domain={{ status: "abstain_out_of_domain", reasons: [
      { code: "abstain_out_of_domain", feature: "HNR", detail: "99 outside training range [8, 33]" }] }} />);
    const el = screen.getByRole("status");
    expect(el).toHaveTextContent(/no prediction made/i);
    expect(el).toHaveTextContent(/HNR: 99 outside training range/);
    expect(el).not.toHaveTextContent(/predicted:|healthy pattern|%/i);
  });
  it("distinguishes invalid input from out-of-domain", () => {
    render(<AbstainNotice domain={{ status: "abstain_invalid_input", reasons: [] }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/not valid/i);
  });
  it("truncates long reason lists and tolerates a missing domain", () => {
    const reasons = Array.from({ length: 9 }, (_, i) => ({ code: "c", feature: `f${i}`, detail: "d" }));
    const { unmount } = render(<AbstainNotice domain={{ reasons }} />);
    expect(screen.getByText(/and 4 more/)).toBeInTheDocument();
    unmount();
    render(<AbstainNotice domain={null} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});
