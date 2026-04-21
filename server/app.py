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

import html
import importlib
import os
import re
from pathlib import Path
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            return parts[1]
    return text


def _render_readme_to_html() -> str:
    try:
        readme_text = README_PATH.read_text(encoding="utf-8")
        readme_text = _strip_front_matter(readme_text)
    except Exception:
        readme_text = "# OfficeAgentEnv\n\nREADME.md not found."

    try:
        md = importlib.import_module("markdown")

        rendered = md.markdown(
            readme_text,
            extensions=["fenced_code", "tables", "toc", "sane_lists"],
        )
    except Exception:
        # Fallback keeps content visible even if markdown package is unavailable.
        escaped = html.escape(readme_text)
        rendered = f"<pre>{escaped}</pre>"

    # Force links in README to open safely in new tab.
    rendered = re.sub(r"<a ", '<a target="_blank" rel="noopener noreferrer" ', rendered)
    return rendered


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
        readme_html = _render_readme_to_html()
        return f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>OfficeAgentEnv</title>
    <style>
        :root {{
            --bg: #f5f7fb;
            --ink: #0f172a;
            --muted: #475569;
            --line: #e2e8f0;
            --card: #ffffff;
            --brand: #0f766e;
            --brand-soft: #ccfbf1;
            --link: #0369a1;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            color: var(--ink);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background:
                radial-gradient(circle at 90% -20%, #dbeafe 0%, rgba(219, 234, 254, 0) 40%),
                radial-gradient(circle at -10% 0%, #dcfce7 0%, rgba(220, 252, 231, 0) 32%),
                var(--bg);
        }}
        .wrap {{
            max-width: 980px;
            margin: 28px auto;
            padding: 0 16px 28px;
        }}
        .hero {{
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
        }}
        .hero h1 {{
            margin: 0;
            font-size: clamp(1.5rem, 2.5vw, 2rem);
        }}
        .hero p {{
            margin: 10px 0 0;
            color: var(--muted);
            line-height: 1.6;
        }}
        .chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }}
        .chip {{
            background: var(--brand-soft);
            border: 1px solid #99f6e4;
            color: #134e4a;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.86rem;
            font-weight: 600;
        }}
        .links {{
            margin-top: 14px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 10px;
        }}
        .links a {{
            text-decoration: none;
            color: var(--ink);
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 11px 12px;
            font-weight: 600;
            transition: all 0.15s ease;
        }}
        .links a:hover {{
            border-color: #93c5fd;
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        }}
        .doc {{
            margin-top: 16px;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
            line-height: 1.72;
        }}
        .doc h1, .doc h2, .doc h3 {{
            line-height: 1.3;
            margin-top: 1.5em;
            margin-bottom: 0.45em;
            scroll-margin-top: 10px;
        }}
        .doc h1:first-child {{ margin-top: 0; }}
        .doc p {{ color: #1e293b; }}
        .doc a {{ color: var(--link); }}
        .doc table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 18px;
            font-size: 0.94rem;
        }}
        .doc th, .doc td {{
            border: 1px solid var(--line);
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }}
        .doc th {{ background: #f8fafc; }}
        .doc pre {{
            overflow-x: auto;
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 10px;
            padding: 12px;
            font-size: 0.9rem;
        }}
        .doc code {{
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.9em;
        }}
        .doc :not(pre) > code {{
            background: #f1f5f9;
            padding: 1px 6px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            color: #0f172a;
        }}
        @media (max-width: 640px) {{
            .hero, .doc {{ padding: 16px; }}
        }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <header class=\"hero\">
            <h1>OfficeAgentEnv</h1>
            <p>Professional environment demo with full project documentation rendered from README.md.</p>
            <div class=\"chips\">
                <span class=\"chip\">Running · v1.0.0</span>
                <span class=\"chip\">OpenEnv Compatible</span>
                <span class=\"chip\">FastAPI + Docker</span>
            </div>
            <div class=\"links\">
                <a href=\"/docs\" target=\"_blank\" rel=\"noopener noreferrer\">Open API Docs</a>
                <a href=\"/openapi.json\" target=\"_blank\" rel=\"noopener noreferrer\">OpenAPI JSON</a>
                <a href=\"/tasks\" target=\"_blank\" rel=\"noopener noreferrer\">View Tasks</a>
                <a href=\"/state\" target=\"_blank\" rel=\"noopener noreferrer\">View State</a>
            </div>
        </header>
        <article class=\"doc\">{readme_html}</article>
    </div>
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
