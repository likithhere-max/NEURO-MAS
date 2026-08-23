from __future__ import annotations

from fastapi import APIRouter

from app.agents.registry import WORKFLOW_STAGES, all_agents

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/status")
def agents_status():
    agents = all_agents()
    return {
        "agents": [
            {
                "name": a.name,
                "capabilities": a.capabilities,
                "reliability_weight": a.reliability_weight,
            }
            for a in agents.values()
        ],
        "workflow_stages": WORKFLOW_STAGES,
    }
