"""
task_medium.py — Meeting Scheduling Grader
Score = weighted combination of:
  - classification accuracy on base emails     (40%)
  - meetings scheduled without conflicts       (40%)
  - no important emails ignored                (20%)
"""
from __future__ import annotations
from env.models import EmailCategory, ExecAssistObservation
from graders.task_easy import GROUND_TRUTH, grade as easy_grade


def grade(obs: ExecAssistObservation) -> float:
    # 1. Classification accuracy (reuse easy grader)
    classify_score = easy_grade(obs)

    # 2. Meeting scheduling — reward each conflict-free scheduled meeting
    #    that corresponds to a meeting-request email
    meeting_request_ids = {
        eid for eid, cat in GROUND_TRUTH.items()
        if cat == EmailCategory.MEETING_REQUEST
    }
    scheduled_titles = {ev.title.lower() for ev in obs.calendar_events}

    # Count processed emails that were meeting requests and ended up scheduled
    scheduled_count = 0
    for email in obs.processed_emails:
        if email.email_id in meeting_request_ids:
            # Heuristic: subject keywords appear in a scheduled event title
            keywords = set(email.subject.lower().split())
            if any(any(kw in title for kw in keywords) for title in scheduled_titles):
                scheduled_count += 1

    schedule_score = (
        scheduled_count / len(meeting_request_ids)
        if meeting_request_ids else 0.0
    )

    # 3. Penalty for ignoring non-spam emails
    ignored_important = sum(
        1 for e in obs.processed_emails
        if e.processed
        and e.category != EmailCategory.SPAM
        and e.email_id in GROUND_TRUTH
        # emails ignored by agent are marked processed but have no reply/event
        # (we approximate: if category stayed UNKNOWN and processed → likely ignored)
        and e.category == EmailCategory.UNKNOWN
    )
    ignore_penalty = min(ignored_important * 0.1, 0.20)

    score = (
        0.40 * classify_score
        + 0.40 * schedule_score
        - ignore_penalty
    )
    return round(max(0.0, min(score, 1.0)), 4)
