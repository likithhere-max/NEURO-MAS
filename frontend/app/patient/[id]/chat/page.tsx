"use client";

import { useEffect, useRef, useState } from "react";
import { usePatient } from "@/components/PatientContext";

const SUGGESTIONS = ["Why did the system consider this condition?", "What factors increased my risk?", "What should I do next?"];

export default function ChatPage() {
  const { chatHistory, chatLoading, sendChatMessage } = usePatient();
  const [input, setInput] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, chatLoading]);

  async function send(text?: string) {
    const msg = (text ?? input).trim();
    if (!msg || chatLoading) return;
    setInput("");
    await sendChatMessage(msg);
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col" style={{ minHeight: "60vh" }}>
      <div className="card mb-3 text-xs text-neuro-muted">
        This assistant explains your current assessment — it does not diagnose or prescribe. It only uses
        information already computed by NEURO-MAS&apos;s agents.
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto card mb-3" style={{ minHeight: "40vh" }}>
        {chatHistory.length === 0 && (
          <div className="text-sm text-neuro-muted">
            Ask about your assessment, risk, or what to do next.
            <div className="flex flex-wrap gap-2 mt-3">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)} className="text-xs border border-neuro-border rounded-full px-3 py-1 hover:border-neuro-accent">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {chatHistory.map((m: any, i: number) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                m.role === "user" ? "bg-neuro-accent text-neuro-bg" : "bg-neuro-panel2 border border-neuro-border"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {chatLoading && <div className="text-xs text-neuro-muted">Assistant is thinking…</div>}
        <div ref={endRef} />
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask about your assessment…"
          className="flex-1 bg-neuro-panel2 border border-neuro-border rounded-lg px-3 py-2 text-sm"
        />
        <button onClick={() => send()} disabled={chatLoading} className="bg-neuro-accent text-neuro-bg font-medium rounded-lg px-4 py-2 text-sm disabled:opacity-50">
          Send
        </button>
      </div>
    </div>
  );
}
