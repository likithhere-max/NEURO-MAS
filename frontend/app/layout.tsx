import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NEURO-MAS | Multi-Agent Neurological Healthcare",
  description: "A Multi-Agent Collaborative System for Autonomous and Personalized Neurological Healthcare",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-neuro-bg">
          <header className="border-b border-neuro-border bg-neuro-panel/60 px-6 py-3 flex items-center justify-between sticky top-0 z-10 backdrop-blur">
            <a href="/" className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-md bg-gradient-to-br from-neuro-accent to-neuro-accent2 flex items-center justify-center text-[11px] font-bold text-neuro-bg">
                NM
              </div>
              <div>
                <div className="text-sm font-semibold leading-none">NEURO-MAS</div>
                <div className="text-[11px] text-neuro-muted leading-none mt-0.5">
                  Multi-Agent · Autonomous · Personalized Neurological Healthcare
                </div>
              </div>
            </a>
            <div className="text-[11px] text-neuro-muted border border-neuro-border rounded-full px-3 py-1">
              PROTOTYPE — for clinician use, not a diagnostic device
            </div>
          </header>
          <main className="px-6 py-6 max-w-7xl mx-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
