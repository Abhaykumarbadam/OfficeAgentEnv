"""Strict grader for the HARD task in OfficeAgentEnv.

Base score is a weighted combination of:

  - classification_accuracy (40%)
  - scheduling_accuracy    (25%)
  - reply_quality          (20%)
  - inbox_completion       (10%)
  - penalty_component      (5%)

After the base score is computed, strict penalties and caps are applied:

  1. Any meeting_request email not scheduled   -> subtract 0.30
  2. Any required reply missing                -> subtract 0.20
  3. Any urgent_task email left unprocessed    -> subtract 0.25
  4. If < 80% of emails processed              -> multiply score by 0.5
  5. If 2 or more of (1)-(3) occur             -> cap score at 0.60

The final score is clamped to [0.0, 1.0].

This design makes the HARD task clearly differentiate weak vs strong agents
while remaining deterministic and based solely on the final environment state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Tuple

from env.models import EmailCategory, ExecAssistObservation
from graders.task_easy import GROUND_TRUTH


def _collect_all_emails(obs: ExecAssistObservation) -> List:
    """Return a deduplicated list of all emails by email_id."""

    by_id = {e.email_id: e for e in (obs.processed_emails + obs.pending_emails)}
    return list(by_id.values())


def _email_is_meeting_request(email) -> bool:
    gt = GROUND_TRUTH.get(email.email_id)
    if gt is not None:
        return gt == EmailCategory.MEETING_REQUEST
    return email.category == EmailCategory.MEETING_REQUEST


def _email_requires_reply(email) -> bool:
    gt = GROUND_TRUTH.get(email.email_id)
    if gt is not None:
        return gt == EmailCategory.GENERAL_QUERY
    return email.category == EmailCategory.GENERAL_QUERY


def _email_is_urgent(email) -> bool:
    gt = GROUND_TRUTH.get(email.email_id)
    if gt is not None:
        return gt == EmailCategory.URGENT_TASK
    return email.category == EmailCategory.URGENT_TASK


def _titles_for_calendar_events(obs: ExecAssistObservation) -> List[str]:
    return [ev.title.lower() for ev in obs.calendar_events]


def _subject_matches_title(subject: str, title: str) -> bool:
    """Heuristic: an overlap between subject keywords and the event title."""

    subject_words = {w for w in subject.lower().split() if w}
    if not subject_words:
        return False
    title_l = title.lower()
    return any(word in title_l for word in subject_words)


def _count_scheduling_conflicts(obs: ExecAssistObservation) -> int:
    """Count calendar events that overlap in time.

    Each conflicting *pair* contributes one to the conflict count.
    """

    events = obs.calendar_events
    n = len(events)
    if n <= 1:
        return 0

    def parse_dt(s: str) -> datetime | None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    conflicts = 0
    for i in range(n):
        a = events[i]
        a_start = parse_dt(a.start_time)
        a_end = parse_dt(a.end_time)
        if not (a_start and a_end):
            continue
        for j in range(i + 1, n):
            b = events[j]
            b_start = parse_dt(b.start_time)
            b_end = parse_dt(b.end_time)
            if not (b_start and b_end):
                continue
            if a_start < b_end and a_end > b_start:
                conflicts += 1
    return conflicts


def compute_classification_score(emails: Iterable) -> float:
    """Accuracy of classifications against ground-truth labels.

    Only emails with ids present in GROUND_TRUTH are counted.
    """

    correct = 0
    total = 0
    for email in emails:
        gt = GROUND_TRUTH.get(email.email_id)
        if gt is None:
            continue
        total += 1
        if email.category == gt:
            correct += 1
    return (correct / total) if total else 0.0


def compute_scheduling_score(emails: Iterable, obs: ExecAssistObservation) -> Tuple[float, int, int]:
    """Score meeting scheduling quality.

    Returns (schedule_score, missing_meeting_count, conflict_count).
    """

    all_emails = list(emails)
    titles = _titles_for_calendar_events(obs)

    meeting_emails = [e for e in all_emails if _email_is_meeting_request(e)]
    if not meeting_emails:
        schedule_score = 1.0
        missing_meetings = 0
    else:
        scheduled_ids = set()
        for em in meeting_emails:
            if any(_subject_matches_title(em.subject, t) for t in titles):
                scheduled_ids.add(em.email_id)
        scheduled_count = len(scheduled_ids)
        missing_meetings = len(meeting_emails) - scheduled_count
        schedule_score = scheduled_count / len(meeting_emails)

    conflicts = _count_scheduling_conflicts(obs)
    return schedule_score, missing_meetings, conflicts


def compute_reply_score(emails: Iterable) -> Tuple[float, int]:
    """Score whether emails that require a reply were processed at all.

    Since reply_text is not available here, we treat a processed email of the
    appropriate type as an attempted reply.
    Returns (score, missing_reply_count).
    """

    reply_needed = [e for e in emails if _email_requires_reply(e)]
    if not reply_needed:
        return 1.0, 0

    replied = [e for e in reply_needed if e.processed]
    missing = len(reply_needed) - len(replied)

    score = len(replied) / len(reply_needed)
    return score, missing


def grade(obs: ExecAssistObservation) -> float:
    """Main grading entrypoint for the HARD task.

    Computes a weighted base score and then applies strict penalties and caps
    as described in the module docstring.
    """

    all_emails = _collect_all_emails(obs)
    total_emails = len(all_emails)
    if total_emails == 0:
        return 0.0

    processed_ids = {e.email_id for e in obs.processed_emails}

    # 1. Core component scores
    classification_accuracy = compute_classification_score(all_emails)
    scheduling_accuracy, missing_meetings, conflict_count = compute_scheduling_score(all_emails, obs)
    reply_quality, missing_replies = compute_reply_score(all_emails)
    inbox_completion = len(obs.processed_emails) / total_emails

    # 2. Penalty component used in the 5% "penalties" weight
    important_pending = 0
    for e in all_emails:
        gt = GROUND_TRUTH.get(e.email_id, e.category)
        if gt in (EmailCategory.MEETING_REQUEST, EmailCategory.URGENT_TASK) and e.email_id not in processed_ids:
            important_pending += 1

    penalty_sum = (
        0.15 * important_pending
        + 0.20 * conflict_count
        + 0.15 * missing_meetings
        + 0.10 * missing_replies
    )
    penalty_component = max(0.0, 1.0 - min(1.0, penalty_sum))

    # 3. Weighted base score (no hard penalties yet)
    base_score = (
        0.40 * classification_accuracy
        + 0.25 * scheduling_accuracy
        + 0.20 * reply_quality
        + 0.10 * inbox_completion
        + 0.05 * penalty_component
    )

    score = base_score

    # ------------------------------------------------------------------
    # Post-score strict penalties and caps
    # ------------------------------------------------------------------

    critical_failures = 0

    # 1. Any meeting_request email not scheduled -> subtract 0.30
    if missing_meetings > 0:
        score -= 0.30
        critical_failures += 1

    # 2. Any required reply missing -> subtract 0.20
    if missing_replies > 0:
        score -= 0.20
        critical_failures += 1

    # 3. Any urgent_task email left unprocessed -> subtract 0.25
    urgent_pending = [e for e in all_emails if _email_is_urgent(e) and not e.processed]
    if urgent_pending:
        score -= 0.25
        critical_failures += 1

    # 4. If < 80% of emails processed -> multiply score by 0.5
    if inbox_completion < 0.8:
        score *= 0.5

    # 5. If 2 or more critical failures -> cap score at 0.60
    if critical_failures >= 2:
        score = min(score, 0.60)

    # Final clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    return round(score, 4)
