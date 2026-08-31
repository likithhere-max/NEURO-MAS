-- Run this in the Supabase SQL editor (same workflow as every prior
-- migration in this project -- no local migration-file system exists yet).
--
-- Adds the two tables needed for the 2-day review sprint features:
--   1. symptom_assessments -- persists each natural-language symptom entry
--      + its Gemini-extracted structured profile, linked to the workflow it
--      triggered.
--   2. chat_messages -- persists the AI Assistant conversation per patient.
--
-- RLS follows the exact pattern already used by every other child table in
-- this project: a subquery against patients.user_id / patients.clinician_id.

create table if not exists symptom_assessments (
    id uuid primary key default gen_random_uuid(),
    patient_id uuid not null references patients(id) on delete cascade,
    workflow_id uuid references workflows(id) on delete set null,
    free_text text not null,
    extracted_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

alter table symptom_assessments enable row level security;

create policy "symptom_assessments_select_own"
    on symptom_assessments for select
    using (
        exists (
            select 1 from patients p
            where p.id = symptom_assessments.patient_id
              and (auth.uid() = p.user_id or auth.uid() = p.clinician_id)
        )
    );

create policy "symptom_assessments_insert_own"
    on symptom_assessments for insert
    with check (
        exists (
            select 1 from patients p
            where p.id = symptom_assessments.patient_id
              and (auth.uid() = p.user_id or auth.uid() = p.clinician_id)
        )
    );

create table if not exists chat_messages (
    id uuid primary key default gen_random_uuid(),
    patient_id uuid not null references patients(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now()
);

alter table chat_messages enable row level security;

create policy "chat_messages_select_own"
    on chat_messages for select
    using (
        exists (
            select 1 from patients p
            where p.id = chat_messages.patient_id
              and (auth.uid() = p.user_id or auth.uid() = p.clinician_id)
        )
    );

create policy "chat_messages_insert_own"
    on chat_messages for insert
    with check (
        exists (
            select 1 from patients p
            where p.id = chat_messages.patient_id
              and (auth.uid() = p.user_id or auth.uid() = p.clinician_id)
        )
    );

-- Helpful for the History tab / chat context builder ordering.
create index if not exists idx_symptom_assessments_patient_created
    on symptom_assessments (patient_id, created_at desc);
create index if not exists idx_chat_messages_patient_created
    on chat_messages (patient_id, created_at asc);
