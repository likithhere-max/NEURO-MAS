"""build_persist_payload (pure) and the RPC persistence path with a fake client (mock: verifies our call shape,
not the SQL in scripts/migration_phase2_persistence.sql, which has NOT been executed)."""
import pytest

import app.core.config as cfg
from app.orchestrator.orchestrator import WorkflowState
from app.schemas.agent import AgentResult, AgentStatus
from app.schemas.patient_state import PatientState
from app.services import repository as repo


def _state(emergency=False):
    s = WorkflowState(workflow_id="00000000-0000-0000-0000-000000000001", patient_id="00000000-0000-0000-0000-0000000000aa",
                      trigger_reason="t", input_payload={}, patient_state=PatientState(patient_id="p"))
    s.agent_results["x"] = AgentResult(agent_name="x", task_id="t", status=AgentStatus.TIMED_OUT, warnings=["agent_timeout:1s"])
    s.audit_events.append({"timestamp": "2026-10-23T00:00:00+00:00", "event": "e", "detail": {"a": 1}})
    s.is_emergency = emergency
    s.timed_out_agents = ["x"]
    s.deadline_exceeded = True
    return s


def test_payload_contains_all_row_groups_and_run_health():
    p = repo.build_persist_payload(_state())
    assert set(p) == {"workflow", "agent_events", "audit_events", "actions", "alerts"}
    rj = p["workflow"]["result_json"]
    assert rj["timed_out_agents"] == ["x"] and rj["deadline_exceeded"] is True
    assert p["agent_events"][0]["status"] == "timed_out"


class _Rpc:
    def __init__(self, data=None, boom=False):
        self.calls, self.data, self.boom = [], data, boom

    def rpc(self, name, args):
        self.calls.append((name, args))
        outer = self

        class E:
            def execute(self_inner):
                if outer.boom:
                    raise RuntimeError("rpc failed")
                class R: pass
                r = R(); r.data = outer.data
                return r
        return E()


def test_rpc_path_sends_one_payload_and_never_touches_tables(monkeypatch):
    monkeypatch.setattr(cfg.get_settings(), "persist_via_rpc", True)
    c = _Rpc(data={"id": "x", "inserted": True})
    out = repo.persist_workflow(c, _state())
    assert out["inserted"] is True
    assert [n for n, _ in c.calls] == ["persist_workflow_atomic"]
    assert c.calls[0][1]["p_payload"]["workflow"]["id"] == "00000000-0000-0000-0000-000000000001"


def test_rpc_failure_propagates_so_callers_can_degrade(monkeypatch):
    monkeypatch.setattr(cfg.get_settings(), "persist_via_rpc", True)
    with pytest.raises(RuntimeError):
        repo.persist_workflow(_Rpc(boom=True), _state())


def test_rpc_empty_result_is_an_error_not_success(monkeypatch):
    monkeypatch.setattr(cfg.get_settings(), "persist_via_rpc", True)
    with pytest.raises(RuntimeError):
        repo.persist_workflow(_Rpc(data=None), _state())
