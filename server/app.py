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
    <title>OfficeAgentEnv - AI Executive Assistant Benchmark</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1a1f35 50%, #16213e 100%);
            color: #e0e0e0;
            line-height: 1.6;
            overflow-x: hidden;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 0 20px; }}
        
        /* HERO SECTION */
        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
            position: relative;
            overflow: hidden;
            padding: 60px 20px;
        }}
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.15), transparent);
            border-radius: 50%;
            pointer-events: none;
            animation: float 6s ease-in-out infinite;
        }}
        .hero::after {{
            content: '';
            position: absolute;
            bottom: -30%;
            left: -10%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.1), transparent);
            border-radius: 50%;
            pointer-events: none;
            animation: float 8s ease-in-out infinite reverse;
        }}
        .hero-content {{
            position: relative;
            z-index: 2;
            max-width: 700px;
        }}
        .hero h1 {{
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hero p {{
            font-size: 1.2rem;
            color: #cbd5e1;
            margin-bottom: 30px;
        }}
        .badge-group {{
            display: flex;
            gap: 12px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .badge {{
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.5);
            color: #60a5fa;
            padding: 8px 16px;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
        }}
        .cta-buttons {{
            display: flex;
            gap: 16px;
            margin-top: 40px;
        }}
        .btn {{
            padding: 14px 32px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
        }}
        .btn-primary:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(59, 130, 246, 0.4);
        }}
        .btn-secondary {{
            background: rgba(255, 255, 255, 0.1);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }}
        .btn-secondary:hover {{
            background: rgba(59, 130, 246, 0.15);
            transform: translateY(-2px);
        }}
        
        /* JUDGING CRITERIA SECTION */
        .section {{
            padding: 80px 20px;
            position: relative;
        }}
        .section-title {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 50px;
            text-align: center;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .criteria-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }}
        .criteria-card {{
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 32px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .criteria-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #60a5fa, transparent);
        }}
        .criteria-card:hover {{
            border-color: rgba(59, 130, 246, 0.5);
            transform: translateY(-8px);
            box-shadow: 0 16px 40px rgba(59, 130, 246, 0.2);
        }}
        .criteria-label {{
            font-size: 0.9rem;
            color: #60a5fa;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .criteria-card h3 {{
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: #e0e0e0;
        }}
        .criteria-percentage {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
        }}
        .criteria-card p {{
            color: #a0aec0;
            line-height: 1.8;
        }}
        
        /* TASKS SECTION */
        .tasks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }}
        .task-card {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 28px;
            transition: all 0.3s ease;
        }}
        .task-card:hover {{
            border-color: rgba(139, 92, 246, 0.6);
            transform: translateY(-6px);
            box-shadow: 0 12px 32px rgba(139, 92, 246, 0.2);
        }}
        .task-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .task-name {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #e0e0e0;
            text-transform: capitalize;
        }}
        .difficulty-badge {{
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .difficulty-easy {{ background: rgba(34, 197, 94, 0.2); color: #86efac; }}
        .difficulty-medium {{ background: rgba(251, 146, 60, 0.2); color: #fdba74; }}
        .difficulty-hard {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; }}
        .task-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 16px 0;
        }}
        .stat {{
            background: rgba(59, 130, 246, 0.1);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.4rem;
            font-weight: 800;
            color: #60a5fa;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .task-desc {{
            color: #a0aec0;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 16px 0;
        }}
        
        /* API ENDPOINTS SECTION */
        .api-section {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 40px;
        }}
        .api-title {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 24px;
            color: #e0e0e0;
        }}
        .endpoint {{
            background: rgba(15, 23, 42, 0.8);
            border-left: 3px solid #60a5fa;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        .endpoint-method {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 700;
            margin-right: 10px;
            font-size: 0.9rem;
        }}
        .method-post {{ background: rgba(59, 130, 246, 0.3); color: #60a5fa; }}
        .method-get {{ background: rgba(34, 197, 94, 0.3); color: #86efac; }}
        .endpoint-path {{
            color: #cbd5e1;
            font-size: 0.95rem;
        }}
        
        /* ANIMATION */
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(30px); }}
        }}
        
        /* DOCUMENTATION SECTION */
        .doc-section {{
            background: rgba(15, 23, 42, 0.3);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 40px;
            margin-top: 80px;
            color: #a0aec0;
        }}
        .doc-section h2 {{
            color: #60a5fa;
            margin-top: 1.5em;
            margin-bottom: 1em;
        }}
        .doc-section table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 0.95rem;
        }}
        .doc-section th, .doc-section td {{
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 12px;
            text-align: left;
        }}
        .doc-section th {{
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            font-weight: 700;
        }}
        .doc-section code {{
            background: rgba(15, 23, 42, 0.8);
            padding: 2px 6px;
            border-radius: 4px;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
        }}
        .doc-section pre {{
            background: rgba(15, 23, 42, 0.8);
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            color: #e0e0e0;
            margin: 1em 0;
        }}
        .doc-section a {{
            color: #60a5fa;
            text-decoration: none;
        }}
        .doc-section a:hover {{
            text-decoration: underline;
        }}
        
        /* FOOTER */
        footer {{
            text-align: center;
            padding: 40px 20px;
            color: #64748b;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            margin-top: 60px;
        }}
        footer p {{ margin: 10px 0; }}
        footer a {{
            color: #60a5fa;
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <!-- HERO -->
    <section class=\"hero\">
        <div class=\"container\">
            <div class=\"hero-content\">
                <h1>OfficeAgentEnv</h1>
                <p>A benchmark for training LLMs to solve real-world executive assistant workflows: email triage, meeting scheduling, and constraint-aware decision-making.</p>
                <div class=\"badge-group\">
                    <span class=\"badge\">🚀 OpenEnv-Compliant</span>
                    <span class=\"badge\">⚡ HF TRL Integrated</span>
                    <span class=\"badge\">🎯 Multi-Task Benchmark</span>
                </div>
                <div class=\"cta-buttons\">
                    <a href=\"#judging-criteria\" class=\"btn btn-primary\">Judging Criteria</a>
                    <a href=\"/docs\" class=\"btn btn-secondary\" target=\"_blank\">API Docs</a>
                </div>
            </div>
        </div>
    </section>


    <!-- TASKS -->
    <section class=\"section\">
        <div class=\"container\">
            <h2 class=\"section-title\">📋 Available Tasks</h2>
            <div class=\"tasks-grid\">
                <div class=\"task-card\">
                    <div class=\"task-header\">
                        <span class=\"task-name\">easy</span>
                        <span class=\"difficulty-badge difficulty-easy\">Easy</span>
                    </div>
                    <div class=\"task-stats\">
                        <div class=\"stat\">
                            <div class=\"stat-value\">10</div>
                            <div class=\"stat-label\">Max Steps</div>
                        </div>
                        <div class=\"stat\">
                            <div class=\"stat-value\">~0.70</div>
                            <div class=\"stat-label\">Baseline</div>
                        </div>
                    </div>
                    <p class=\"task-desc\">Deterministic classification of 5 emails into correct categories (meeting_request, urgent_task, spam, general_query).</p>
                </div>
                <div class=\"task-card\">
                    <div class=\"task-header\">
                        <span class=\"task-name\">medium</span>
                        <span class=\"difficulty-badge difficulty-medium\">Medium</span>
                    </div>
                    <div class=\"task-stats\">
                        <div class=\"stat\">
                            <div class=\"stat-value\">15</div>
                            <div class=\"stat-label\">Max Steps</div>
                        </div>
                        <div class=\"stat\">
                            <div class=\"stat-value\">~0.50</div>
                            <div class=\"stat-label\">Baseline</div>
                        </div>
                    </div>
                    <p class=\"task-desc\">Mixed inbox triage with classification + conflict-aware meeting scheduling. Tests planning and constraint reasoning.</p>
                </div>
                <div class=\"task-card\">
                    <div class=\"task-header\">
                        <span class=\"task-name\">hard</span>
                        <span class=\"difficulty-badge difficulty-hard\">Hard</span>
                    </div>
                    <div class=\"task-stats\">
                        <div class=\"stat\">
                            <div class=\"stat-value\">12</div>
                            <div class=\"stat-label\">Max Steps</div>
                        </div>
                        <div class=\"stat\">
                            <div class=\"stat-value\">~0.38</div>
                            <div class=\"stat-label\">Baseline</div>
                        </div>
                    </div>
                    <p class=\"task-desc\">Full assistant workflow: classify, reply, schedule, and ignore spam. The ultimate test of multi-step reasoning.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- API QUICK REFERENCE -->
    <section class=\"section\">
        <div class=\"container\">
            <h2 class=\"section-title\">⚙️ API Endpoints</h2>
            <div class=\"api-section\">
                <div class=\"api-title\">Quick Reference</div>
                <div class=\"endpoint\">
                    <span class=\"endpoint-method method-post\">POST</span>
                    <span class=\"endpoint-path\">/reset</span>
                    <div style=\"color: #94a3b8; margin-top: 8px; font-size: 0.9rem;\">Reset environment for task. Request: {{\"task\": \"easy\"|\"medium\"|\"hard\", \"seed\": 42}}</div>
                </div>
                <div class=\"endpoint\">
                    <span class=\"endpoint-method method-post\">POST</span>
                    <span class=\"endpoint-path\">/step</span>
                    <div style=\"color: #94a3b8; margin-top: 8px; font-size: 0.9rem;\">Execute one action: classify_email, reply_email, schedule_meeting, or ignore_email</div>
                </div>
                <div class=\"endpoint\">
                    <span class=\"endpoint-method method-get\">GET</span>
                    <span class=\"endpoint-path\">/state</span>
                    <div style=\"color: #94a3b8; margin-top: 8px; font-size: 0.9rem;\">Get current environment state (pending_emails, calendar_events, step count)</div>
                </div>
                <div class=\"endpoint\">
                    <span class=\"endpoint-method method-get\">GET</span>
                    <span class=\"endpoint-path\">/tasks</span>
                    <div style=\"color: #94a3b8; margin-top: 8px; font-size: 0.9rem;\">List all available tasks with difficulty, max_steps, and descriptions</div>
                </div>
                <div class=\"endpoint\">
                    <span class=\"endpoint-method method-post\">POST</span>
                    <span class=\"endpoint-path\">/grade</span>
                    <div style=\"color: #94a3b8; margin-top: 8px; font-size: 0.9rem;\">Score the current episode. Returns score (0-1) for the task</div>
                </div>
            </div>
        </div>
    </section>

    <!-- FULL DOCUMENTATION -->
    <section class=\"section\">
        <div class=\"container\">
            <h2 class=\"section-title\">📚 Full Documentation</h2>
            <div class=\"doc-section\">
                {readme_html}
            </div>
        </div>
    </section>

    <!-- FOOTER -->
    <footer>
        <p>OfficeAgentEnv • OpenEnv Benchmark for Executive Assistant Workflows</p>
        <p>Powered by <a href=\"https://huggingface.co/spaces\" target=\"_blank\">Hugging Face Spaces</a> • <a href=\"/docs\" target=\"_blank\">API Docs</a> • <a href=\"/tasks\" target=\"_blank\">Tasks</a></p>
    </footer>
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
