"use client";

export default function TimelinePanel({ events }: { events: any[] }) {
  return (
    <div className="card">
      <h2 className="font-semibold mb-3">Workflow Timeline</h2>
      {(!events || events.length === 0) && <div className="text-sm text-neuro-muted">No events yet.</div>}
      <ol className="relative border-l border-neuro-border ml-2 space-y-4">
        {events?.map((e, i) => (
          <li key={i} className="ml-4">
            <div className="absolute -ml-[21px] mt-1.5 h-2 w-2 rounded-full bg-neuro-accent" />
            <div className="text-xs text-neuro-muted">{new Date(e.timestamp).toLocaleTimeString()}</div>
            <div className="text-sm font-medium">{e.event.replaceAll("_", " ")}</div>
            {e.detail && Object.keys(e.detail).length > 0 && (
              <div className="text-xs text-neuro-muted mt-0.5 break-all">{JSON.stringify(e.detail)}</div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
