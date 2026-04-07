"""task_medium.py — deterministic behavior-aware medium grader.

Score combines:
  - classification accuracy on known base emails (40%)
  - meeting-request emails handled via schedule action (40%)
  - avoiding ignore action on important emails (20%)
"""
from __future__ import annotations
from env.models import EmailCategory, ExecAssistObservation
from graders.task_easy import GROUND_TRUTH, grade as easy_grade


def grade(obs: ExecAssistObservation) -> float:
    # 1) Classification accuracy on deterministic base set.
    classify_score = easy_grade(obs)

    # 2) Scheduling quality based on explicit action trace.
    meeting_request_ids = {
        eid for eid, cat in GROUND_TRUTH.items()
        if cat == EmailCategory.MEETING_REQUEST
    }
    processed_by_id = {e.email_id: e for e in obs.processed_emails}
    scheduled_count = sum(
        1
        for eid in meeting_request_ids
        if processed_by_id.get(eid) is not None
        and processed_by_id[eid].resolution == "schedule"
    )

    schedule_score = (
        scheduled_count / len(meeting_request_ids)
        if meeting_request_ids else 0.0
    )

    # 3) Penalize ignoring important known emails.
    ignored_important = sum(
        1
        for e in obs.processed_emails
        if e.email_id in GROUND_TRUTH
        and GROUND_TRUTH[e.email_id] != EmailCategory.SPAM
        and e.resolution == "ignore"
    )
    ignore_score = max(0.0, 1.0 - min(ignored_important * 0.5, 1.0))

    score = (
        0.40 * classify_score
        + 0.40 * schedule_score
        + 0.20 * ignore_score
    )
    return round(max(0.0, min(score, 1.0)), 4)
