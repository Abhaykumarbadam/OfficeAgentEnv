from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    MEETING_REQUEST = "meeting_request"
    URGENT_TASK     = "urgent_task"
    SPAM            = "spam"
    GENERAL_QUERY   = "general_query"
    UNKNOWN         = "unknown"


class ActionType(str, Enum):
    CLASSIFY_EMAIL   = "classify_email"
    REPLY_EMAIL      = "reply_email"
    SCHEDULE_MEETING = "schedule_meeting"
    IGNORE_EMAIL     = "ignore_email"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Email(BaseModel):
    email_id:   str
    sender:     str
    subject:    str
    body:       str
    timestamp:  str
    category:   EmailCategory = EmailCategory.UNKNOWN   # ground truth (hidden from agent)
    processed:  bool = False


class CalendarEvent(BaseModel):
    event_id:     str
    title:        str
    start_time:   str   # ISO-like "YYYY-MM-DD HH:MM"
    end_time:     str
    participants: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class ExecAssistAction(BaseModel):
    action_type: ActionType
    email_id:    str
    # for classify_email
    category:    Optional[EmailCategory] = None
    # for reply_email
    reply_text:  Optional[str] = None
    # for schedule_meeting
    meeting_title:       Optional[str] = None
    meeting_start_time:  Optional[str] = None   # "YYYY-MM-DD HH:MM"
    meeting_end_time:    Optional[str] = None
    participants:        Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class ExecAssistObservation(BaseModel):
    pending_emails:    List[Email]
    processed_emails:  List[Email]
    calendar_events:   List[CalendarEvent]
    last_action_result: Optional[str] = None
    current_step:       int = 0
    task_name:          str = ""


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class ExecAssistReward(BaseModel):
    value:       float = 0.0
    reason:      str   = ""
    breakdown:   Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: ExecAssistObservation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)
