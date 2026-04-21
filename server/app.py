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
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def root() -> str:
        return """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>OfficeAgentEnv</title>
    <style>
        :root {
            --bg1: #f7fafc;
            --bg2: #edf2f7;
            --card: #ffffff;
            --text: #1a202c;
            --muted: #4a5568;
            --accent: #0b7285;
            --accent-2: #0891b2;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: \"Segoe UI\", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text);
            background: radial-gradient(circle at 20% 10%, #d9f3ff 0%, transparent 35%),
                                    radial-gradient(circle at 80% 15%, #d7ffe6 0%, transparent 30%),
                                    linear-gradient(180deg, var(--bg1), var(--bg2));
            min-height: 100vh;
            display: grid;
            place-items: center;
            padding: 24px;
        }
        .card {
            width: min(860px, 100%);
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        }
        h1 {
            margin: 0 0 8px 0;
            font-size: clamp(1.6rem, 2.6vw, 2.2rem);
            letter-spacing: 0.2px;
        }
        p {
            margin: 0;
            color: var(--muted);
            line-height: 1.55;
        }
        .status {
            margin-top: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: #ecfeff;
            color: #0f766e;
            border: 1px solid #a5f3fc;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .grid {
            margin-top: 22px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        a.tile {
            text-decoration: none;
            color: inherit;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            background: #fff;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        a.tile:hover {
            transform: translateY(-2px);
            border-color: #a5b4fc;
            box-shadow: 0 10px 20px rgba(2, 6, 23, 0.08);
        }
        .tile-title {
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 6px;
        }
        .tile-text {
            color: var(--muted);
            font-size: 0.92rem;
        }
        code {
            display: block;
            margin-top: 18px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #f8fafc;
            padding: 12px;
            font-family: Consolas, \"Courier New\", monospace;
            overflow-x: auto;
            white-space: pre;
            color: #334155;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <main class=\"card\">
        <h1>OfficeAgentEnv</h1>
        <p>OpenEnv-compatible executive assistant environment for reset, step, grading, and task evaluation.</p>
        <div class=\"status\">Running · v1.0.0</div>

        <section class=\"grid\">
            <a class=\"tile\" href=\"/docs\" target=\"_blank\" rel=\"noopener noreferrer\">
                <div class=\"tile-title\">API Docs</div>
                <div class=\"tile-text\">Interactive Swagger UI for all endpoints.</div>
            </a>
            <a class=\"tile\" href=\"/openapi.json\" target=\"_blank\" rel=\"noopener noreferrer\">
                <div class=\"tile-title\">OpenAPI Spec</div>
                <div class=\"tile-text\">Machine-readable API contract in JSON.</div>
            </a>
            <a class=\"tile\" href=\"/tasks\" target=\"_blank\" rel=\"noopener noreferrer\">
                <div class=\"tile-title\">Tasks</div>
                <div class=\"tile-text\">Available benchmark tasks: easy, medium, hard.</div>
            </a>
            <a class=\"tile\" href=\"/state\" target=\"_blank\" rel=\"noopener noreferrer\">
                <div class=\"tile-title\">State</div>
                <div class=\"tile-text\">Inspect current environment state snapshot.</div>
            </a>
        </section>

        <code>POST /reset  -> initialize task session
POST /step   -> submit action and get reward
POST /grade  -> compute task score</code>
    </main>
</body>
</html>
"""


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
                "max_steps": 12,
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

    obs = env._make_obs_internal()  # internal state for deterministic grading
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
