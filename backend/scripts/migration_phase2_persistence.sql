-- Phase 2 persistence migration -- NOT EXECUTED by the assistant (no database access in the build environment).
-- Apply to a TEST project first; then set PERSIST_VIA_RPC=true.
--
-- 1. Idempotency: a client-supplied Idempotency-Key is mapped (uuid5(patient_id, key)) to workflows.id, so the
--    primary key already prevents duplicate runs; the extra column stores the key for audit.
alter table public.workflows add column if not exists idempotency_key text;
create unique index if not exists workflows_patient_idem_uidx
    on public.workflows (patient_id, idempotency_key) where idempotency_key is not null;

-- 2. All-or-nothing persistence of one workflow. SECURITY INVOKER: row-level security of the calling user applies
--    to every insert, exactly as it does for the REST calls this replaces.
create or replace function public.persist_workflow_atomic(p_payload jsonb)
returns jsonb
language plpgsql
security invoker
as $$
declare
    wf jsonb := p_payload->'workflow';
    inserted_id uuid;
begin
    insert into public.workflows (id, patient_id, trigger_reason, status, is_emergency, started_at, completed_at, result_json)
    values ((wf->>'id')::uuid, (wf->>'patient_id')::uuid, wf->>'trigger_reason', wf->>'status',
            (wf->>'is_emergency')::boolean, (wf->>'started_at')::timestamptz, (wf->>'completed_at')::timestamptz,
            (wf->'result_json') || jsonb_build_object('persist_state', 'complete'))
    on conflict (id) do nothing
    returning id into inserted_id;

    if inserted_id is null then
        -- replay: the run is already stored; write nothing else
        return jsonb_build_object('id', wf->>'id', 'inserted', false);
    end if;

    insert into public.agent_event_log (workflow_id, patient_id, agent_name, status, confidence, result_json)
    select (e->>'workflow_id')::uuid, (e->>'patient_id')::uuid, e->>'agent_name', e->>'status',
           (e->>'confidence')::double precision, e->'result_json'
    from jsonb_array_elements(coalesce(p_payload->'agent_events', '[]'::jsonb)) e;

    insert into public.audit_events (workflow_id, patient_id, event, detail_json, "timestamp")
    select (e->>'workflow_id')::uuid, (e->>'patient_id')::uuid, e->>'event', e->'detail_json', (e->>'timestamp')::timestamptz
    from jsonb_array_elements(coalesce(p_payload->'audit_events', '[]'::jsonb)) e;

    insert into public.actions (id, patient_id, workflow_id, action_type, reason, evidence_json, risk_level, autonomy_level,
                                requires_human_approval, approved_by, status, safety_reasons_json)
    select (e->>'id')::uuid, (e->>'patient_id')::uuid, (e->>'workflow_id')::uuid, e->>'action_type', e->>'reason',
           e->'evidence_json', e->>'risk_level', (e->>'autonomy_level')::int, (e->>'requires_human_approval')::boolean,
           nullif(e->>'approved_by', '')::uuid, e->>'status', e->'safety_reasons_json'
    from jsonb_array_elements(coalesce(p_payload->'actions', '[]'::jsonb)) e;

    insert into public.alerts (patient_id, workflow_id, severity, reason, evidence_json, status)
    select (e->>'patient_id')::uuid, (e->>'workflow_id')::uuid, e->>'severity', e->>'reason', e->'evidence_json', e->>'status'
    from jsonb_array_elements(coalesce(p_payload->'alerts', '[]'::jsonb)) e;

    return jsonb_build_object('id', wf->>'id', 'inserted', true);
end;
$$;
-- Column types above are taken from the application's insert payloads; verify them against the live schema before
-- applying (the repository does not contain the original DDL).
