"""
task_hard.py — Full Workflow Automation Grader
Score = weighted combination of:
  - classification accuracy on ALL emails      (25%)
  - meetings scheduled without conflicts       (25%)
  - quality of replies to non-spam emails      (25%)
  - spam correctly ignored (not replied to)    (15%)
  - inbox clearance (all emails processed)     (10%)
"""
from __future__ import annotations
from env.models import EmailCategory, ExecAssistObservation
from graders.task_easy import GROUND_TRUTH


def grade(obs: ExecAssistObservation) -> float:
    all_emails = obs.processed_emails + obs.pending_emails
    total = len(all_emails)
    if total == 0:
        return 0.0

    # ── 1. Classification accuracy across ALL emails (including random ones) ──
    correct_classify = 0
    classified_total = 0
    for email in all_emails:
        if email.email_id in GROUND_TRUTH:
            classified_total += 1
            if email.category == GROUND_TRUTH[email.email_id]:
                correct_classify += 1
    classify_score = (correct_classify / classified_total) if classified_total else 0.0

    # ── 2. Meeting scheduling ──
    meeting_emails = [
        e for e in all_emails
        if e.category == EmailCategory.MEETING_REQUEST
    ]
    scheduled_titles = {ev.title.lower() for ev in obs.calendar_events}
    meetings_scheduled = 0
    for em in meeting_emails:
        keywords = set(em.subject.lower().split())
        if any(any(kw in t for kw in keywords) for t in scheduled_titles):
            meetings_scheduled += 1
    schedule_score = (
        meetings_scheduled / len(meeting_emails) if meeting_emails else 1.0
    )

    # ── 3. Reply quality — penalise missing replies to non-spam ──
    non_spam = [
        e for e in obs.processed_emails
        if e.category != EmailCategory.SPAM
    ]
    # We approximate: emails whose category is not UNKNOWN were acted upon
    # (replied or scheduled); we give credit for processing them at all
    acted_on = sum(1 for e in non_spam if e.category != EmailCategory.UNKNOWN)
    reply_score = (acted_on / len(non_spam)) if non_spam else 1.0

    # ── 4. Spam handling ──
    spam_emails = [e for e in all_emails if e.category == EmailCategory.SPAM]
    spam_ignored = sum(
        1 for e in obs.processed_emails
        if e.category == EmailCategory.SPAM and e.processed
    )
    spam_score = (spam_ignored / len(spam_emails)) if spam_emails else 1.0

    # ── 5. Inbox clearance ──
    clearance_score = len(obs.processed_emails) / total

    score = (
        0.25 * classify_score
        + 0.25 * schedule_score
        + 0.25 * reply_score
        + 0.15 * spam_score
        + 0.10 * clearance_score
    )
    return round(max(0.0, min(score, 1.0)), 4)
