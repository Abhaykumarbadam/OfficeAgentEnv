"""
environment.py
Core OpenEnv-compliant environment for OfficeAgentEnv.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from env.email_data import build_calendar_events, get_emails_for_task
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
# Reward constants
# ---------------------------------------------------------------------------

# Core dense rewards
R_CORRECT_CLASSIFY   =  0.30
R_WRONG_CLASSIFY     = -0.20
R_GOOD_REPLY         =  0.20
R_BAD_REPLY          = -0.10
R_SCHEDULE_OK        =  0.30
R_SCHEDULE_CONFLICT  = -0.25
R_IGNORE_SPAM        =  0.10
R_IGNORE_IMPORTANT   = -0.15

# Per-step penalty encourages shorter solutions
R_STEP_PENALTY       = -0.02

# Additional nuanced penalties / bonuses
R_INVALID_EMAIL      = -0.10   # invalid email id
R_DUPLICATE_ACTION   = -0.10   # acting again on an already processed email
R_OUT_OF_HOURS       = -0.20   # meeting outside working hours
R_TOO_SHORT_MEETING  = -0.10   # meeting shorter than 15 minutes
R_LOW_QUALITY_REPLY  = -0.10   # reply text is too short / uninformative
R_BONUS_COMPLETE     =  0.20   # bonus for fully clearing the inbox

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
        reward, result_msg = self._apply_action(action)

        # Per-step shaping penalty
        reward += R_STEP_PENALTY

        # Bonus for fully and (approximately) correctly clearing the inbox.
        # We award this when the inbox becomes empty as a result of this step.
        if not self._pending:
            reward += R_BONUS_COMPLETE
            result_msg += " All emails processed; completion bonus applied."

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
        email = self._find_email(action.email_id)

        # Distinguish between invalid ids and duplicate processing attempts.
        if email is None:
            if any(e.email_id == action.email_id for e in self._processed):
                return (
                    R_DUPLICATE_ACTION,
                    f"Email '{action.email_id}' was already processed; duplicate action penalized.",
                )
            return (
                R_INVALID_EMAIL,
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

        return (0.0, "Unknown action type.")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_classify(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        if action.category is None:
            return (-0.05, "classify_email requires a 'category' field.")

        correct = email.category == action.category
        reward  = R_CORRECT_CLASSIFY if correct else R_WRONG_CLASSIFY

        email.category  = action.category   # agent's classification persists
        email.processed = True
        email.resolution = "classify"
        self._move_to_processed(email)

        status = "CORRECT" if correct else "WRONG"
        return (reward, f"[{status}] Classified '{email.email_id}' as '{action.category.value}'.")

    def _do_reply(self, email: Email, action: ExecAssistAction) -> Tuple[float, str]:
        if not action.reply_text:
            return (R_LOW_QUALITY_REPLY, "reply_email requires non-empty 'reply_text'.")

        quality = _reply_quality(action.reply_text, email)
        if quality <= 0.2:
            reward = R_LOW_QUALITY_REPLY
            msg_detail = "low-quality reply (too short or generic)."
        else:
            reward = R_GOOD_REPLY * quality
            msg_detail = "reply accepted."

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
            return (-0.10, "schedule_meeting requires 'meeting_start_time' and 'meeting_end_time'.")

        parsed_start = _parse_dt(start)
        parsed_end = _parse_dt(end)
        if not parsed_start or not parsed_end:
            return (-0.10, "Invalid datetime format for meeting_start_time or meeting_end_time.")

        # Enforce working hours (09:00–18:00)
        if not _within_working_hours(parsed_start, parsed_end):
            return (
                R_OUT_OF_HOURS,
                "Requested meeting time is outside working hours (09:00–18:00).",
            )

        # Enforce minimum duration of 15 minutes
        if (parsed_end - parsed_start).total_seconds() < 15 * 60:
            return (
                R_TOO_SHORT_MEETING,
                "Requested meeting is shorter than 15 minutes.",
            )

        if check_schedule_conflict(self._calendar, start, end):
            return (R_SCHEDULE_CONFLICT, f"Scheduling conflict for '{start}' – '{end}'.")

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

        return (R_SCHEDULE_OK, f"Meeting '{title}' scheduled at {start}.")

    def _do_ignore(self, email: Email) -> Tuple[float, str]:
        intents = classify_intent(email)
        is_spam = intents["is_spam"]
        reward  = R_IGNORE_SPAM if is_spam else R_IGNORE_IMPORTANT

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
        return ExecAssistObservation(
            pending_emails=deepcopy(self._pending),
            processed_emails=deepcopy(self._processed),
            calendar_events=deepcopy(self._calendar),
            last_action_result=self._last_result,
            current_step=self._step_count,
            task_name=self.task_name,
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

        return ExecAssistObservation(
            pending_emails=pending,
            processed_emails=processed,
            calendar_events=deepcopy(self._calendar),
            last_action_result=self._last_result,
            current_step=self._step_count,
            task_name=self.task_name,
        )
