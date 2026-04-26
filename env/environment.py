"""
environment.py
Core OpenEnv-compliant environment for OfficeAgentEnv.
"""
from __future__ import annotations

import uuid
import random
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from env.email_data import build_calendar_events, build_random_emails, get_emails_for_task
from env.models import (
    ActionType,
    CalendarEvent,
    Email,
    EmailCategory,
    ExecAssistAction,
    ExecAssistObservation,
    StepResult,
)


# ---------------------------------------------------------------------------
# Event-based reward parameters
# ---------------------------------------------------------------------------

# Reward is modeled as:
# r_t = w_s * e_success + w_q * e_quality + w_e * e_eff - w_v * e_violation - w_d * e_delayed
# where each event term is binary/bounded and final reward is clipped to [-1, 1].
W_SUCCESS   = 0.55
W_QUALITY   = 0.25
W_EFFICIENCY = 0.20
W_VIOLATION = 0.70
W_DELAYED   = 0.60


def event_reward(
    *,
    success: float = 0.0,
    quality: float = 0.0,
    efficiency: float = 0.0,
    violation: float = 0.0,
    delayed: float = 0.0,
) -> float:
    raw = (
        W_SUCCESS * max(0.0, min(1.0, success))
        + W_QUALITY * max(0.0, min(1.0, quality))
        + W_EFFICIENCY * max(0.0, min(1.0, efficiency))
        - W_VIOLATION * max(0.0, min(1.0, violation))
        - W_DELAYED * max(0.0, min(1.0, delayed))
    )
    return max(-1.0, min(1.0, raw))


def normalized_episode_score(total_reward: float, max_steps: int) -> float:
    """Map episode return to a fair normalized score in [0, 100]."""
    # With per-step clipping in [-1, 1], theoretical episode bounds are [-max_steps, max_steps].
    g_min = float(-max_steps)
    g_max = float(max_steps)
    denom = max(g_max - g_min, 1e-8)
    score = 100.0 * ((total_reward - g_min) / denom)
    return max(0.0, min(100.0, score))

MAX_STEPS: Dict[str, int] = {
    "easy":   10,
    "medium": 15,
    "hard":   12,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def check_schedule_conflict(events: List[CalendarEvent], start: str, end: str) -> bool:
    """Return True if the interval [start, end] overlaps any existing event.

    This mirrors the grader's notion of a scheduling conflict and is used
    both for reward shaping and for hard-task evaluation.
    """
    new_start = _parse_dt(start)
    new_end   = _parse_dt(end)
    if new_start is None or new_end is None:
        return False
    for ev in events:
        ev_start = _parse_dt(ev.start_time)
        ev_end   = _parse_dt(ev.end_time)
        if ev_start and ev_end:
            if new_start < ev_end and new_end > ev_start:
                return True
    return False


def _reply_quality(reply_text: str, email: Email) -> float:
    """
    Heuristic: reward is higher for longer, relevant replies.
    Returns a multiplier in [0, 1].
    """
    if not is_valid_reply(reply_text):
        return 0.0
    score = min(len(reply_text) / 200, 1.0)  # length up to 200 chars → 1.0
    # Bonus if reply references sender name or subject keywords
    keywords = set(email.subject.lower().split() + [email.sender.split("@")[0].lower()])
    hits = sum(1 for kw in keywords if kw in reply_text.lower())
    score = min(score + hits * 0.05, 1.0)
    return score


def is_valid_reply(reply_text: str) -> bool:
    """Basic check that a reply is non-trivial.

    Used to penalize very short or empty replies. The threshold is chosen
    to align with the reward spec (low-quality reply < 15 chars).
    """

    if not reply_text:
        return False
    return len(reply_text.strip()) >= 15


def classify_intent(email: Email) -> Dict[str, bool]:
    """Derive simple intent flags from the email category.

    These flags are redundant with EmailCategory but make it easier for
    agents to reason about the inbox when surfaced in explanations.
    """

    is_meeting_request = email.category == EmailCategory.MEETING_REQUEST
    is_spam = email.category == EmailCategory.SPAM
    is_urgent = email.category == EmailCategory.URGENT_TASK
    requires_reply = email.category == EmailCategory.GENERAL_QUERY
    return {
        "is_meeting_request": is_meeting_request,
        "is_spam": is_spam,
        "is_urgent": is_urgent,
        "requires_reply": requires_reply,
    }


def _within_working_hours(start: datetime, end: datetime) -> bool:
    """Check that a meeting is within standard working hours (09:00–18:00)."""

    if start.date() != end.date():
        return False
    if start.hour < 9 or end.hour > 18:
        return False
    return True


def _count_events_for_day(events: List[CalendarEvent], day: datetime) -> int:
    """Count scheduled events for the same calendar day."""
    count = 0
    for ev in events:
        ev_start = _parse_dt(ev.start_time)
        if ev_start and ev_start.date() == day.date():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ExecAssistEnv:
    """
    OpenEnv-compliant Executive Assistant environment.

    Tasks
    -----
    easy   – classify 5 deterministic emails correctly
    medium – classify + schedule meetings from a mixed inbox
    hard   – full workflow: classify, reply, schedule, ignore spam
    """

    VALID_TASKS = ("easy", "medium", "hard")

    def __init__(self, task_name: str = "easy", seed: int = 42):
        assert task_name in self.VALID_TASKS, f"task must be one of {self.VALID_TASKS}"
        self.task_name  = task_name
        self.seed       = seed
        self._max_steps = MAX_STEPS[task_name]

        # runtime state (initialised in reset)
        self._pending:   List[Email]         = []
        self._processed: List[Email]         = []
        self._calendar:  List[CalendarEvent] = []
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._last_result: str = ""
        self.world_state: Dict[str, Any] = {
            "projects": [
                {"id": "P1", "status": "on_track", "deadline": 5},
                {"id": "P2", "status": "delayed", "deadline": 3},
            ],
            "team_load": {
                "engineering": 2,
                "sales": 1,
            },
            "client_satisfaction": 0.75,
        }
        self.delayed_events: List[Dict[str, Any]] = []
        self.debug_mode: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> ExecAssistObservation:
        self._pending    = get_emails_for_task(self.task_name, seed=self.seed)
        self._processed  = []
        self._calendar   = build_calendar_events()
        self._step_count = 0
        self._total_reward = 0.0
        self._done       = False
        self._last_result = "Environment reset. Process your inbox."
        return self._make_obs_public()

    def step(self, action: ExecAssistAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._make_obs_public(),
                reward=0.0,
                done=True,
                info={"warning": "Episode already finished."},
            )

        self._step_count += 1
        delayed_signal = 0.0
        remaining_events: List[Dict[str, Any]] = []
        for event in self.delayed_events:
            if event.get("trigger_step") == self._step_count:
                delayed_signal += float(event.get("delayed", 0.0))
            else:
                remaining_events.append(event)
        self.delayed_events = remaining_events
        if self.debug_mode and delayed_signal != 0.0:
            print(f"Delayed penalty event triggered: {delayed_signal}")

        if random.random() < 0.3:
            new_email = build_random_emails(n=1, seed=self.seed + self._step_count)
            if new_email:
                self._pending.append(new_email[0])

        reward, result_msg = self._apply_action(action)

        # Per-step shaping: small efficiency term for shorter trajectories.
        step_efficiency = max(0.0, 1.0 - min(self._step_count / max(self._max_steps, 1), 1.0))
        reward += event_reward(efficiency=0.15 * step_efficiency)
        reward += event_reward(delayed=delayed_signal)

        # Bonus for fully and (approximately) correctly clearing the inbox.
        # We award this when the inbox becomes empty as a result of this step.
        if not self._pending:
            reward += event_reward(success=0.4, quality=0.4)
            result_msg += " All emails processed; completion bonus applied."

        base_reward = reward
        team_load_penalty = 0.05 * sum(self.world_state.get("team_load", {}).values())
        time_penalty = 0.01 * self._step_count
        reward = base_reward - time_penalty - team_load_penalty

        self._total_reward += reward
        self._last_result = result_msg

        # Episode ends when inbox is empty OR max steps reached
        if not self._pending or self._step_count >= self._max_steps:
            self._done = True

        return StepResult(
            observation=self._make_obs_public(),
            reward=round(reward, 4),
            done=self._done,
            info={
                "step": self._step_count,
                "total_reward": round(self._total_reward, 4),
                "normalized_score_0_100": round(
                    normalized_episode_score(self._total_reward, self._max_steps), 2
                ),
                "emails_remaining": len(self._pending),
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_name":        self.task_name,
            "step_count":       self._step_count,
            "max_steps":        self._max_steps,
            "total_reward":     round(self._total_reward, 4),
            "done":             self._done,
            "emails_pending":   len(self._pending),
            "emails_processed": len(self._processed),
            "calendar_events":  len(self._calendar),
        }

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: ExecAssistAction) -> Tuple[float, str]:
        if self.task_name == "easy" and action.action_type != ActionType.CLASSIFY_EMAIL:
            return (
                event_reward(violation=0.55),
                "Easy task allows classify_email only; non-classification action penalized.",
            )

        if action.action_type == ActionType.QUERY_STATUS:
            return self._do_query_status(action)

        if action.action_type == ActionType.UPDATE_PROJECT:
            return self._do_update_project(action)

        email = self._find_email(action.email_id)

        # Distinguish between invalid ids and duplicate processing attempts.
        if email is None:
            if any(e.email_id == action.email_id for e in self._processed):
                return (
                    event_reward(violation=0.35),
                    f"Email '{action.email_id}' was already processed; duplicate action penalized.",
                )
            return (
                event_reward(violation=0.30),
                f"Email '{action.email_id}' not found in pending inbox.",
            )

        if action.action_type == ActionType.CLASSIFY_EMAIL:
            return self._do_classify(email, action)

        elif action.action_type == ActionType.REPLY_EMAIL:
            return self._do_reply(email, action)

        elif action.action_type == ActionType.SCHEDULE_MEETING:
            return self._do_schedule(email, action)

        elif action.action_type == ActionType.IGNORE_EMAIL:
            return self._do_ignore(email)

        elif action.action_type == ActionType.ASSIGN_TASK:
            return self._do_assign_task(email, action)

        return (-0.05, f"Unknown action type '{action.action_type}'.")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_classify(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        if action.category is None:
            return (event_reward(violation=0.20), "classify_email requires a 'category' field.")

        correct = email.category == action.category
        reward = (
            event_reward(success=1.0, quality=1.0, efficiency=0.4)
            if correct
            else event_reward(violation=0.45)
        )

        email.category  = action.category   # agent's classification persists
        email.processed = True
        email.resolution = "classify"
        self._move_to_processed(email)

        status = "CORRECT" if correct else "WRONG"
        return (reward, f"[{status}] Classified '{email.email_id}' as '{action.category.value}'.")

    def _do_reply(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        if not action.reply_text:
            return (event_reward(violation=0.25), "reply_email requires non-empty 'reply_text'.")

        quality = _reply_quality(action.reply_text, email)
        if quality <= 0.2:
            reward = event_reward(violation=0.20)
            msg_detail = "low-quality reply (too short or generic)."
        else:
            reward = event_reward(success=0.7, quality=quality, efficiency=0.2)
            msg_detail = "reply accepted."

        if email.category != EmailCategory.GENERAL_QUERY:
            reward += event_reward(violation=0.20)
            msg_detail += " wrong-intent penalty applied."

        # Replying without explicit classification should not grant label credit.
        email.category = EmailCategory.UNKNOWN
        email.processed = True
        email.resolution = "reply"
        self._move_to_processed(email)

        return (
            reward,
            f"Replied to '{email.email_id}' (quality={quality:.2f}, {msg_detail})",
        )

    def _do_schedule(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        start = action.meeting_start_time
        end   = action.meeting_end_time
        title = action.meeting_title or email.subject

        if not start or not end:
            return (
                event_reward(violation=0.25),
                "schedule_meeting requires 'meeting_start_time' and 'meeting_end_time'.",
            )

        parsed_start = _parse_dt(start)
        parsed_end = _parse_dt(end)
        if not parsed_start or not parsed_end:
            return (event_reward(violation=0.25), "Invalid datetime format for meeting_start_time or meeting_end_time.")

        # Enforce working hours (09:00–18:00)
        if not _within_working_hours(parsed_start, parsed_end):
            return (
                event_reward(violation=0.35),
                "Requested meeting time is outside working hours (09:00–18:00).",
            )

        # Enforce minimum duration of 15 minutes
        if (parsed_end - parsed_start).total_seconds() < 15 * 60:
            return (
                event_reward(violation=0.25),
                "Requested meeting is shorter than 15 minutes.",
            )

        if check_schedule_conflict(self._calendar, start, end):
            return (event_reward(violation=0.40), f"Scheduling conflict for '{start}' – '{end}'.")

        reward = event_reward(success=1.0, quality=0.9, efficiency=0.3)
        msg_suffix = ""
        if email.category != EmailCategory.MEETING_REQUEST:
            reward += event_reward(violation=0.22)
            msg_suffix += " Wrong-intent scheduling penalty applied."

        # Long-horizon constraint: discourage over-scheduling a single day.
        if _count_events_for_day(self._calendar, parsed_start) >= 4:
            reward += event_reward(violation=0.30)
            msg_suffix += " Daily scheduling limit penalty applied."

        new_event = CalendarEvent(
            event_id=str(uuid.uuid4())[:8],
            title=title,
            start_time=start,
            end_time=end,
            participants=action.participants or [email.sender],
        )
        self._calendar.append(new_event)
        # Scheduling without explicit classification should not grant label credit.
        email.category = EmailCategory.UNKNOWN
        email.processed = True
        email.resolution = "schedule"
        self._move_to_processed(email)

        return (reward, f"Meeting '{title}' scheduled at {start}.{msg_suffix}")

    def _do_ignore(self, email: Email) -> Tuple[float, str]:
        intents = classify_intent(email)
        is_spam = intents["is_spam"]
        reward = event_reward(success=0.6, quality=0.8, efficiency=0.3) if is_spam else event_reward(violation=0.35)
        if not is_spam:
            self.delayed_events.append(
                {
                    "trigger_step": self._step_count + 2,
                    "delayed": 0.45,
                }
            )

        # Ignoring without explicit classification should not grant label credit.
        email.category = EmailCategory.UNKNOWN
        email.processed = True
        email.resolution = "ignore"
        self._move_to_processed(email)
        if is_spam:
            msg = f"Correctly ignored spam email '{email.email_id}'."
        else:
            msg = f"Ignored important email '{email.email_id}' (may hurt your score)."
        return (reward, msg)

    def _do_assign_task(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        team = (action.team or "engineering").strip().lower()
        loads = self.world_state.setdefault("team_load", {})
        loads[team] = int(loads.get(team, 0)) + 1
        if loads[team] > 5:
            reward = event_reward(violation=0.35)
        else:
            reward = event_reward(success=0.6, quality=0.6, efficiency=0.4)

        email.category = EmailCategory.UNKNOWN
        email.processed = True
        email.resolution = "assign"
        self._move_to_processed(email)
        return (reward, f"Assigned task from '{email.email_id}' to team '{team}'.")

    def _do_query_status(self, action: ExecAssistAction) -> Tuple[float, str]:
        _ = self.world_state.get("projects", [])
        _ = self.world_state.get("client_satisfaction", 0.75)
        return (event_reward(efficiency=0.2), "Status queried successfully.")

    def _do_update_project(self, action: ExecAssistAction) -> Tuple[float, str]:
        project_id = (action.project_id or "").strip()
        project_status = (action.project_status or "").strip()
        valid_statuses = {"on_track", "delayed", "blocked", "completed"}
        if not project_id or not project_status or project_status not in valid_statuses:
            return (event_reward(violation=0.25), "Invalid project update request.")

        projects = self.world_state.get("projects", [])
        for proj in projects:
            if proj.get("id") == project_id:
                proj["status"] = project_status
                if project_status == "completed":
                    self.world_state["client_satisfaction"] = min(
                        1.0, float(self.world_state.get("client_satisfaction", 0.75)) + 0.05
                    )
                return (
                    event_reward(success=0.8, quality=0.7, efficiency=0.3),
                    f"Project '{project_id}' updated to '{project_status}'.",
                )
        return (event_reward(violation=0.20), f"Project '{project_id}' not found.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_email(self, email_id: str) -> Optional[Email]:
        for e in self._pending:
            if e.email_id == email_id:
                return e
        return None

    def _move_to_processed(self, email: Email) -> None:
        self._pending   = [e for e in self._pending if e.email_id != email.email_id]
        self._processed.append(email)

    def _make_obs_internal(self) -> ExecAssistObservation:
        projects = self.world_state.get("projects", [])
        compact_world_state = {
            "projects": [
                {"id": p.get("id"), "status": p.get("status"), "deadline": p.get("deadline")}
                for p in projects
            ],
            "team_load": dict(self.world_state.get("team_load", {})),
            "client_satisfaction": self.world_state.get("client_satisfaction", 0.75),
        }
        return ExecAssistObservation(
            pending_emails=deepcopy(self._pending),
            processed_emails=deepcopy(self._processed),
            calendar_events=deepcopy(self._calendar),
            last_action_result=self._last_result,
            current_step=self._step_count,
            task_name=self.task_name,
            world_state=compact_world_state,
        )

    def _make_obs_public(self) -> ExecAssistObservation:
        """Build an observation safe for agent consumption.

        Ground-truth labels are hidden by replacing category values with UNKNOWN.
        This prevents reward/grader leakage via observation payloads.
        """

        pending = deepcopy(self._pending)
        processed = deepcopy(self._processed)

        for email in pending:
            email.category = EmailCategory.UNKNOWN
        for email in processed:
            email.category = EmailCategory.UNKNOWN

        projects = self.world_state.get("projects", [])
        compact_world_state = {
            "projects": [
                {"id": p.get("id"), "status": p.get("status"), "deadline": p.get("deadline")}
                for p in projects
            ],
            "team_load": dict(self.world_state.get("team_load", {})),
            "client_satisfaction": self.world_state.get("client_satisfaction", 0.75),
        }

        return ExecAssistObservation(
            pending_emails=pending,
            processed_emails=processed,
            calendar_events=deepcopy(self._calendar),
            last_action_result=self._last_result,
            current_step=self._step_count,
            task_name=self.task_name,
            world_state=compact_world_state,
        )
