"""Orchestration mechanics: dependency order, bounded concurrency, deadlines, cancellation, fail-closed (Phase 2, D3)."""
import asyncio
import time

import pytest

import app.orchestrator.orchestrator as orch
from app.agents.registry import AGENT_DEPENDENCIES, WORKFLOW_STAGES, ancestors, get_agent, validate_dependencies
from app.schemas.agent import AgentResult, AgentStatus
from app.schemas.patient_state import PatientState


class _S:
    def __init__(self, per=5.0, deadline=30.0, conc=4):
        self.agent_timeout_seconds, self.workflow_deadline_seconds, self.max_agent_concurrency = per, deadline, conc


def _settings(monkeypatch, **kw):
    s = _S(**kw)
    monkeypatch.setattr(orch, "get_settings", lambda: s)


class Recorder:
    def __init__(self):
        self.start, self.end, self.running, self.max_running = {}, {}, 0, 0
        self.upstream = {}


def _patch_all(monkeypatch, rec: Recorder, delay=0.05, hang=(), fail=(), abstain=(), payload=None):
    payload = payload or {}
    for name in AGENT_DEPENDENCIES:
        agent = get_agent(name)

        async def fake(task, state, _n=name):
            rec.start[_n] = time.perf_counter()
            rec.upstream[_n] = set(task.input_payload.get("upstream_results", {}))
            rec.running += 1
            rec.max_running = max(rec.max_running, rec.running)
            try:
                if _n in hang:
                    await asyncio.sleep(60)
                await asyncio.sleep(delay)
                if _n in fail:
                    raise RuntimeError("boom")
                if _n in abstain:
                    return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.ABSTAINED)
                return AgentResult(agent_name=_n, task_id=task.task_id, status=AgentStatus.COMPLETED,
                                   raw_payload=payload.get(_n, {}))
            finally:
                rec.running -= 1
                rec.end[_n] = time.perf_counter()

        monkeypatch.setattr(agent, "execute", fake)


async def _run(**kw):
    return await orch.run_workflow(patient_id="p", trigger_reason="t", patient_state=PatientState(patient_id="p"),
                                   input_payload=kw.get("input_payload", {"symptoms": [{"name": "mild headache"}]}),
                                   **{k: v for k, v in kw.items() if k != "input_payload"})


# ---------------------------------------------------------------- graph
def test_dependency_graph_is_acyclic_and_complete():
    layers = validate_dependencies(AGENT_DEPENDENCIES)
    assert sorted(a for layer in layers for a in layer) == sorted(AGENT_DEPENDENCIES)
    assert layers == WORKFLOW_STAGES and layers[0] == ["data_intake_agent"]


def test_cycle_and_unknown_dependency_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        validate_dependencies({"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        validate_dependencies({"a": ["zzz"]})


def test_ancestors_are_transitive():
    assert {"data_intake_agent", "neuro_diagnostic_agent"} <= ancestors("knowledge_agent")
    assert "longitudinal_agent" not in ancestors("signal_analysis_agent")


# ---------------------------------------------------------------- ordering & concurrency
@pytest.mark.asyncio
async def test_dependents_start_only_after_dependencies_finish(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)
    await _run()
    for name, deps in AGENT_DEPENDENCIES.items():
        for d in deps:
            assert rec.start[name] >= rec.end[d], f"{name} started before {d} finished"


@pytest.mark.asyncio
async def test_independent_agents_overlap_and_concurrency_is_bounded(monkeypatch):
    rec = Recorder()
    _settings(monkeypatch, conc=4)
    _patch_all(monkeypatch, rec, delay=0.1)
    await _run()
    assert rec.max_running >= 3                      # diagnostic/signal/longitudinal/medication overlap
    rec2 = Recorder()
    _settings(monkeypatch, conc=1)
    _patch_all(monkeypatch, rec2, delay=0.02)
    await _run()
    assert rec2.max_running == 1                     # semaphore bound honoured


@pytest.mark.asyncio
async def test_concurrent_run_is_faster_than_serial_sum(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec, delay=0.1)
    t0 = time.perf_counter()
    await _run()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1 * len(AGENT_DEPENDENCIES) * 0.8      # measured, not asserted from theory


@pytest.mark.asyncio
async def test_upstream_view_contains_only_declared_ancestors(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)
    await _run()
    for name in AGENT_DEPENDENCIES:
        assert rec.upstream[name] <= ancestors(name)
    assert "longitudinal_agent" not in rec.upstream["signal_analysis_agent"]
    assert "neuro_diagnostic_agent" in rec.upstream["knowledge_agent"]


# ---------------------------------------------------------------- failure, typed statuses
@pytest.mark.asyncio
async def test_failure_of_one_agent_is_isolated_and_dependents_still_run(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec, fail=("neuro_diagnostic_agent",))
    r = await _run()
    assert r.agent_results["neuro_diagnostic_agent"].status == "failed"
    assert r.agent_results["knowledge_agent"].status == "completed"      # dependency ordering, not success, gates it
    assert r.degraded and r.failed_agents == ["neuro_diagnostic_agent"]
    assert r.consensus is None                                            # no evidence -> no consensus invented


@pytest.mark.asyncio
async def test_abstained_agent_is_not_a_failure(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec, abstain=("signal_analysis_agent",))
    r = await _run()
    assert r.agent_results["signal_analysis_agent"].status == "abstained"
    assert r.degraded is False and r.failed_agents == []


@pytest.mark.asyncio
async def test_results_mapping_order_is_deterministic(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)
    r = await _run()
    assert list(r.agent_results) == [n for n in AGENT_DEPENDENCIES]


# ---------------------------------------------------------------- deadlines
@pytest.mark.asyncio
async def test_per_agent_timeout_is_typed_timed_out(monkeypatch):
    _settings(monkeypatch, per=0.1)
    rec = Recorder()
    _patch_all(monkeypatch, rec, hang=("longitudinal_agent",))
    r = await _run()
    assert r.agent_results["longitudinal_agent"].status == "timed_out"
    assert "agent_timeout" in r.agent_results["longitudinal_agent"].warnings[0]
    assert r.timed_out_agents == ["longitudinal_agent"] and r.deadline_exceeded is False
    assert r.agent_results["risk_stratification_agent"].status == "completed"


@pytest.mark.asyncio
async def test_global_deadline_returns_a_structured_partial_result_quickly(monkeypatch):
    _settings(monkeypatch, per=30.0, deadline=0.4)
    rec = Recorder()
    _patch_all(monkeypatch, rec, hang=("neuro_diagnostic_agent", "signal_analysis_agent"))
    t0 = time.perf_counter()
    r = await _run()
    assert time.perf_counter() - t0 < 2.0
    assert r.deadline_exceeded and r.degraded
    assert r.agent_results["neuro_diagnostic_agent"].status == "timed_out"
    assert r.agent_results["neuro_diagnostic_agent"].warnings == ["global_deadline_exceeded"]
    # every registered agent has exactly one terminal result: nothing silently dropped
    assert set(r.agent_results) == set(AGENT_DEPENDENCIES)
    assert r.status == "completed" and r.completed_at is not None


@pytest.mark.asyncio
async def test_agents_not_started_before_deadline_are_timed_out_not_skipped(monkeypatch):
    _settings(monkeypatch, per=30.0, deadline=0.15)
    rec = Recorder()
    _patch_all(monkeypatch, rec, hang=("data_intake_agent",))
    r = await _run()
    late = [a for a, res in r.agent_results.items() if res.status == "timed_out"]
    assert "data_intake_agent" in late and len(late) >= 6
    assert any("before_start" in w for res in r.agent_results.values() for w in res.warnings)


# ---------------------------------------------------------------- emergency preservation
@pytest.mark.asyncio
async def test_emergency_agent_timeout_fails_closed(monkeypatch):
    _settings(monkeypatch, per=0.1)
    rec = Recorder()
    _patch_all(monkeypatch, rec, hang=("emergency_escalation_agent",))
    r = await _run()
    assert r.agent_results["emergency_escalation_agent"].status == "timed_out"
    assert r.is_emergency is True
    assert any(a.action_type == "emergency_escalation" for a in r.proposed_actions)


@pytest.mark.asyncio
async def test_emergency_agent_abstaining_also_fails_closed(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec, abstain=("emergency_escalation_agent",))
    r = await _run()
    assert r.is_emergency is True and r.agent_results["emergency_escalation_agent"].raw_payload["failed_closed"] is True


@pytest.mark.asyncio
async def test_prescan_emergency_survives_every_other_agent_failing(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec, fail=tuple(n for n in AGENT_DEPENDENCIES if n != "emergency_escalation_agent"))
    r = await _run(input_payload={"symptoms": [{"name": "seizure", "free_text": "status epilepticus, seizure not stopping"}]})
    assert r.is_emergency and any(a.action_type == "emergency_escalation" for a in r.proposed_actions)


# ---------------------------------------------------------------- cancellation
@pytest.mark.asyncio
async def test_cancellation_cancels_running_agents_and_leaves_no_orphans(monkeypatch):
    _settings(monkeypatch, per=30.0, deadline=60.0)
    cancelled = []
    for name in AGENT_DEPENDENCIES:
        agent = get_agent(name)

        async def slow(task, state, _n=name):
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(_n)
                raise
        monkeypatch.setattr(agent, "execute", slow)
    t = asyncio.create_task(_run())
    await asyncio.sleep(0.2)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert "data_intake_agent" in cancelled
    leftovers = [x for x in asyncio.all_tasks() if x.get_name().startswith("agent:") and not x.done()]
    assert leftovers == []


# ---------------------------------------------------------------- checkpoints & idempotent id
@pytest.mark.asyncio
async def test_prescan_checkpoint_fires_before_agents_run(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)
    seen = {}

    async def cb(state, name):
        seen["name"], seen["emergency"], seen["agents_started"] = name, state.is_emergency, len(rec.start)

    await _run(on_checkpoint=cb, input_payload={"symptoms": [{"name": "x", "free_text": "he collapsed and is not responding"}]})
    assert seen == {"name": "prescan", "emergency": True, "agents_started": 0}


@pytest.mark.asyncio
async def test_failing_checkpoint_never_blocks_the_workflow(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)

    async def cb(state, name):
        raise RuntimeError("db down")
    r = await _run(on_checkpoint=cb)
    assert r.status == "completed"
    assert any(e["event"] == "checkpoint_failed" for e in r.audit_events)


@pytest.mark.asyncio
async def test_caller_supplied_workflow_id_is_used(monkeypatch):
    _settings(monkeypatch)
    rec = Recorder()
    _patch_all(monkeypatch, rec)
    r = await _run(workflow_id="wf-fixed-123")
    assert r.workflow_id == "wf-fixed-123"
