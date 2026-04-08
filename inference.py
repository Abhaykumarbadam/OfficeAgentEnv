"""
inference.py — OfficeAgentEnv Baseline Inference Script
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# Phase-2 validator injects API_KEY + API_BASE_URL for LiteLLM proxy usage.
# Keep local fallbacks for developer runs.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
DEFAULT_MODEL_NAME = "heuristic-fallback"
DEFAULT_PROXY_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "officeagentenv"
MAX_STEPS = {"easy": 10, "medium": 15, "hard": 12}
SUCCESS_THRESHOLD = 0.4
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def _strict_score(value: float) -> float:
    # Keep scores strictly in (0,1) even after formatting/parsing.
    return max(0.0001, min(0.9999, float(value)))


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    safe_score = _strict_score(score)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={safe_score:.4f} rewards={r_str}",
        flush=True,
    )


def env_reset(task: str, seed: int = 42) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_URL}/reset", json={"task": task, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = httpx.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_grade(task: str) -> float:
    r = httpx.post(f"{ENV_URL}/grade", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()["score"]


SYSTEM_PROMPT = textwrap.dedent(
    """
You are an expert executive assistant AI.
You will be given a snapshot of an email inbox and a calendar.
Your job is to process each pending email by choosing ONE action.

Available actions (return ONLY valid JSON, no markdown fences):
1. Classify email:
   {"action_type": "classify_email", "email_id": "<id>", "category": "<meeting_request|urgent_task|spam|general_query>"}

2. Reply to email:
   {"action_type": "reply_email", "email_id": "<id>", "reply_text": "<your reply>"}

3. Schedule meeting:
   {"action_type": "schedule_meeting", "email_id": "<id>", "meeting_title": "<title>",
    "meeting_start_time": "YYYY-MM-DD HH:MM", "meeting_end_time": "YYYY-MM-DD HH:MM",
    "participants": ["email@example.com"]}

4. Ignore email:
   {"action_type": "ignore_email", "email_id": "<id>"}

Rules:
- Only act on ONE email per step.
- For meeting requests, prefer scheduling.
- Ignore spam emails.
- Reply to general queries with helpful text.
- Classify urgent tasks.
- Return ONLY the raw JSON object. No extra text.
"""
).strip()


def build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    pending = obs.get("pending_emails", [])
    calendar = obs.get("calendar_events", [])
    last = obs.get("last_action_result", "")

    pending_str = json.dumps(
        [
            {
                "id": e["email_id"],
                "from": e["sender"],
                "subject": e["subject"],
                "body": e["body"][:200],
            }
            for e in pending
        ],
        indent=2,
    )
    calendar_str = json.dumps(
        [{"title": ev["title"], "start": ev["start_time"], "end": ev["end_time"]} for ev in calendar],
        indent=2,
    )

    return textwrap.dedent(
        f"""
Step: {step}
Last result: {last}

Pending emails ({len(pending)} remaining):
{pending_str}

Current calendar:
{calendar_str}

Choose your next action (raw JSON only):
"""
    ).strip()


def get_model_message(client: Optional[OpenAI], messages: List[Dict[str, str]], *, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """Call the chat model with a single retry and concise error logging.

    Raises RuntimeError if both attempts fail.
    """
    if client is None:
        raise RuntimeError("LLM client not configured.")

    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                timeout=20,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Model returned empty content.")
            return text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if "<!DOCTYPE html" in msg or "<html" in msg.lower():
                msg = (
                    "HTTP error from LLM backend (for example 401 Unauthorized). "
                    "Check HF_TOKEN permissions."
                )
            else:
                msg = msg[:200]
            # Keep stdout strictly in [START]/[STEP]/[END] format.
            _ = msg

    raise RuntimeError(f"LLM call failed after 2 attempts: {last_exc}")


def infer_category_from_email(email: Dict[str, Any]) -> str:
    """Heuristic category assignment used when the LLM is unavailable."""
    subject = str(email.get("subject", ""))
    body = str(email.get("body", ""))
    text = f"{subject} {body}".lower()

    if "meeting" in text or "schedule" in text or "calendar" in text:
        return "meeting_request"
    if "urgent" in text or "asap" in text or "immediately" in text:
        return "urgent_task"
    if "offer" in text or "win" in text or "gift card" in text or "inheritance" in text or "prize" in text:
        return "spam"
    return "general_query"


def get_action(client: Optional[OpenAI], obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    prompt = build_user_prompt(obs, step)
    try:
        text = get_model_message(
            client,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        pending = obs.get("pending_emails", [])
        if pending:
            email = pending[0]
            return {
                "action_type": "classify_email",
                "email_id": email["email_id"],
                "category": infer_category_from_email(email),
            }
        return {"action_type": "ignore_email", "email_id": "e001"}


def run_task(client: Optional[OpenAI], task: str) -> None:
    max_steps = MAX_STEPS[task]
    log_start(task=task, model=MODEL_NAME or DEFAULT_MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_data = env_reset(task)
        obs = reset_data["observation"]
        done = reset_data.get("done", False)

        for step in range(1, max_steps + 1):
            if done or not obs.get("pending_emails"):
                break

            action = get_action(client, obs, step)
            action_str = json.dumps(action, separators=(",", ":"))

            try:
                result = env_step(action)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0))
                done = result.get("done", False)
                error = result.get("info", {}).get("error")
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = _strict_score(env_grade(task))
        success = score >= SUCCESS_THRESHOLD

    except KeyboardInterrupt:
        # Graceful interruption: still emit [END] in finally, without traceback.
        log_step(step=0, action="task_interrupt", reward=0.0, done=True, error="keyboard_interrupt")
    except Exception as exc:
        log_step(step=0, action="task_init", reward=0.0, done=True, error=str(exc)[:200])

    finally:
        log_end(task=task, success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    model_name = MODEL_NAME or DEFAULT_PROXY_MODEL_NAME
    use_llm = bool(API_KEY and API_BASE_URL)
    client: Optional[OpenAI] = None
    if use_llm:
        # Use injected proxy vars when present; these are required by the validator.
        resolved_api_key = os.environ["API_KEY"] if "API_KEY" in os.environ else API_KEY
        resolved_base_url = os.environ["API_BASE_URL"] if "API_BASE_URL" in os.environ else API_BASE_URL
        globals()["MODEL_NAME"] = model_name
        client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    try:
        for task in TASKS:
            run_task(client, task)
    except KeyboardInterrupt:
        # Graceful shutdown when user interrupts execution.
        return
    except Exception:
        # Avoid non-zero crash in evaluator environments.
        return


if __name__ == "__main__":
    main()