"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

function tabs(id: string) {
  return [
    { href: `/patient/${id}`, label: "Dashboard" },
    { href: `/patient/${id}/assessment`, label: "Symptom Assessment" },
    { href: `/patient/${id}/analysis`, label: "Analysis" },
    { href: `/patient/${id}/risk`, label: "Risk" },
    { href: `/patient/${id}/longitudinal`, label: "Longitudinal" },
    { href: `/patient/${id}/evidence`, label: "Evidence" },
    { href: `/patient/${id}/next-steps`, label: "Next Steps" },
    { href: `/patient/${id}/chat`, label: "AI Assistant" },
    { href: `/patient/${id}/history`, label: "History" },
    { href: `/patient/${id}/profile`, label: "Profile" },
  ];
}

export default function TabNav({ patientId }: { patientId: string }) {
  const pathname = usePathname();
  const items = tabs(patientId);
  const dashboardHref = `/patient/${patientId}`;

  return (
    <nav className="flex flex-wrap gap-1 border-b border-neuro-border mb-6 pb-px overflow-x-auto">
      {items.map((t) => {
        const active = t.href === dashboardHref ? pathname === t.href : pathname?.startsWith(t.href);
        return (
          <Link
            key={t.href}
            href={t.href}
            className={`px-3 py-2 text-sm whitespace-nowrap border-b-2 -mb-px transition-colors ${
              active ? "border-neuro-accent text-neuro-text font-medium" : "border-transparent text-neuro-muted hover:text-neuro-text"
            }`}
          >
            {t.label}
          </Link>
        );
      })}
    </nav>
  );
}
