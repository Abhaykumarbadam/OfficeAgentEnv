"""
server/app.py
FastAPI server exposing the OpenEnv HTTP interface:
  POST /reset
  POST /step
  GET  /state
  GET  /tasks
  POST /grade
"""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import ExecAssistEnv
from env.models import ExecAssistAction, ExecAssistObservation, StepResult
import graders.task_easy   as grader_easy
import graders.task_medium as grader_medium
import graders.task_hard   as grader_hard


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OfficeAgentEnv",
    description="OpenEnv-compliant executive assistant environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global env registry (one env per task)
# ---------------------------------------------------------------------------

ENVS: Dict[str, ExecAssistEnv] = {
    "easy":   ExecAssistEnv(task_name="easy",   seed=42),
    "medium": ExecAssistEnv(task_name="medium", seed=42),
    "hard":   ExecAssistEnv(task_name="hard",   seed=42),
}

GRADERS = {
    "easy":   grader_easy.grade,
    "medium": grader_medium.grade,
    "hard":   grader_hard.grade,
}

_active_task: str = "easy"
_active_obs: ExecAssistObservation | None = None


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    task_name: str | None = None
    seed: int = 42


class GradeRequest(BaseModel):
    task: str = "easy"
    task_name: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "env": "OfficeAgentEnv", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> Dict[str, Any]:
    global _active_task, _active_obs

    if req is None:
        req = ResetRequest()

    requested_task = req.task_name or req.task

    if requested_task not in ENVS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{requested_task}'. Choose from: {list(ENVS)}")

    _active_task = requested_task
    ENVS[requested_task] = ExecAssistEnv(task_name=requested_task, seed=req.seed)
    obs = ENVS[requested_task].reset()
    _active_obs = obs
    return {"observation": obs.model_dump(), "done": False, "reward": 0.0, "info": {}}


@app.post("/step")
def step(action: ExecAssistAction) -> Dict[str, Any]:
    global _active_obs

    env = ENVS.get(_active_task)
    if env is None:
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")

    result: StepResult = env.step(action)
    _active_obs = result.observation
    return result.model_dump()


@app.get("/state")
def state() -> Dict[str, Any]:
    env = ENVS.get(_active_task)
    if env is None:
        raise HTTPException(status_code=400, detail="No active environment.")
    return env.state()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Classify 5 deterministic emails into correct categories.",
                "difficulty": "easy",
                "max_steps": 10,
            },
            {
                "name": "medium",
                "description": "Classify emails AND schedule conflict-free meetings from a mixed inbox.",
                "difficulty": "medium",
                "max_steps": 15,
            },
            {
                "name": "hard",
                "description": "Full workflow: classify, reply, schedule, and ignore spam across a noisy inbox.",
                "difficulty": "hard",
                "max_steps": 20,
            },
        ]
    }


@app.post("/grade")
def grade(req: GradeRequest | None = None) -> Dict[str, Any]:
    if req is None:
        req = GradeRequest()

    requested_task = req.task_name or req.task
    env = ENVS.get(requested_task)
    if env is None:
        raise HTTPException(status_code=400, detail=f"Unknown task '{requested_task}'.")

    obs = env._make_obs()  # get current obs without stepping
    grader = GRADERS[requested_task]
    score = grader(obs)

    return {
        "task":  requested_task,
        "score": score,
        "state": env.state(),
    }


def main() -> None:
    """CLI entrypoint required by OpenEnv validation for server launch."""

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
