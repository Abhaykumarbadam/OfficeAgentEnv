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

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "officeagentenv"
MAX_STEPS = {"easy": 10, "medium": 15, "hard": 20}
SUCCESS_THRESHOLD = 0.4
TASKS = ["easy", "medium", "hard"]


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)


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


def get_model_message(client: OpenAI, messages: List[Dict[str, str]], *, max_tokens: int = 300, temperature: float = 0.2) -> str:
    """Call the chat model with a single retry and concise error logging.

    Raises RuntimeError if both attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
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
                    "Check HF_TOKEN or OPENAI_API_KEY permissions."
                )
            else:
                msg = msg[:200]
            print(f"[DEBUG] LLM call error (attempt {attempt + 1}/2): {msg}", flush=True)

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


def get_action(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    prompt = build_user_prompt(obs, step)
    try:
        text = get_model_message(
            client,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        print("[DEBUG] Falling back to heuristic action due to LLM failure.", flush=True)
        pending = obs.get("pending_emails", [])
        if pending:
            email = pending[0]
            return {
                "action_type": "classify_email",
                "email_id": email["email_id"],
                "category": infer_category_from_email(email),
            }
        return {"action_type": "ignore_email", "email_id": "e001"}


def run_task(client: OpenAI, task: str) -> None:
    max_steps = MAX_STEPS[task]
    log_start(task=task, model=MODEL_NAME)

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

        score = env_grade(task)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not API_KEY:
        raise ValueError("Missing API key. Set HF_TOKEN or OPENAI_API_KEY.")

    print(f"[DEBUG] API key loaded: {'yes' if API_KEY else 'no'}", flush=True)
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()