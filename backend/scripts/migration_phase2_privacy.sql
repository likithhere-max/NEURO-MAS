-- Phase 2 privacy migration -- NOT EXECUTED by the assistant. Apply to a TEST project first.
-- Consent to external (LLM) processing, recorded per patient. Default false = fail closed.
alter table public.patients add column if not exists external_ai_consent boolean not null default false;
alter table public.patients add column if not exists external_ai_consent_at timestamptz;
alter table public.patients add column if not exists external_ai_consent_by uuid;
-- Existing RLS policies on public.patients decide who may update these columns; review them: a patient or the
-- treating clinician should be able to, nobody else. (The original policies are not in this repository.)
