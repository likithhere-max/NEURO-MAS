-- Phase 2 notification outbox -- NOT EXECUTED by the assistant. Apply to a TEST project first.
-- Server-owned table (written with the service-role key). RLS is enabled with NO policies, so end users and anonymous
-- callers can neither read nor write it. Holds ids/states only: no recipient address, no message body.
create table if not exists public.notifications (
    id uuid primary key,
    idempotency_key text not null unique,
    action_id uuid not null,
    workflow_id uuid not null,
    patient_id uuid not null,
    channel text not null,
    action_type text not null,
    recipient_ref text,
    state text not null check (state in ('proposed','approved','queued','sent','delivered','failed','acknowledged','escalated')),
    attempts int not null default 0,
    last_error_class text,
    provider_message_id text,
    history jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);
alter table public.notifications enable row level security;
-- Verify column types against the live schema (workflows/actions ids) before applying.
