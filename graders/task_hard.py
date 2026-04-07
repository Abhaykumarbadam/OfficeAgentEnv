"""Strict behavior-aware HARD grader.

Design goals:
- Score quality across the full hard inbox (including random emails).
- Reward correct intent-to-action mapping, not just task completion.
- Penalize critical mistakes (e.g., ignoring urgent or mishandling spam).
- Keep deterministic final score in [0.0, 1.0].
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Dict, Iterable, List

from env.models import EmailCategory, ExecAssistObservation
from graders.task_easy import GROUND_TRUTH

HARD_GROUND_TRUTH: Dict[str, EmailCategory] = {
    "h001": EmailCategory.MEETING_REQUEST,
    "h002": EmailCategory.SPAM,
    "h003": EmailCategory.URGENT_TASK,
    "h004": EmailCategory.MEETING_REQUEST,
    "h005": EmailCategory.SPAM,
    "h006": EmailCategory.GENERAL_QUERY,
}

ALL_KNOWN_TRUTH: Dict[str, EmailCategory] = {**GROUND_TRUTH, **HARD_GROUND_TRUTH}


def _collect_all_emails(obs: ExecAssistObservation) -> List:
    """Return deduplicated emails by email_id."""

    by_id = {e.email_id: e for e in (obs.processed_emails + obs.pending_emails)}
    return list(by_id.values())


def _infer_expected_category(email) -> EmailCategory:
    """Deterministic category inference for emails not in fixed truth maps."""

    mapped = ALL_KNOWN_TRUTH.get(email.email_id)
    if mapped is not None:
        return mapped

    text = f"{email.subject} {email.body}".lower()
    if any(k in text for k in ["urgent", "critical", "outage", "p1", "immediate", "asap"]):
        return EmailCategory.URGENT_TASK
    if any(k in text for k in ["meeting", "schedule", "calendar", "sync", "demo", "review"]):
        return EmailCategory.MEETING_REQUEST
    if any(k in text for k in ["gift card", "discount", "offer", "prize", "inheritance", "verify account"]):
        return EmailCategory.SPAM
    return EmailCategory.GENERAL_QUERY


def _expected_resolution_for_category(category: EmailCategory) -> str:
    if category == EmailCategory.MEETING_REQUEST:
        return "schedule"
    if category == EmailCategory.SPAM:
        return "ignore"
    if category == EmailCategory.GENERAL_QUERY:
        return "reply"
    # urgent tasks are typically triaged via classify
    return "classify"


def _count_conflicts(obs: ExecAssistObservation) -> int:
    events = obs.calendar_events
    if len(events) <= 1:
        return 0

    def parse_dt(s: str) -> datetime | None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    conflicts = 0
    for i in range(len(events)):
        a_start = parse_dt(events[i].start_time)
        a_end = parse_dt(events[i].end_time)
        if not (a_start and a_end):
            continue
        for j in range(i + 1, len(events)):
            b_start = parse_dt(events[j].start_time)
            b_end = parse_dt(events[j].end_time)
            if not (b_start and b_end):
                continue
            if a_start < b_end and a_end > b_start:
                conflicts += 1
    return conflicts


def compute_classification_score(emails: Iterable) -> float:
    correct = 0
    total = 0
    for email in emails:
        total += 1
        gt = _infer_expected_category(email)
        # Classification credit is awarded only for explicit classify action.
        if email.resolution == "classify" and email.category == gt:
            correct += 1
    return (correct / total) if total else 0.0


def _action_accuracy(emails: Iterable) -> float:
    all_emails = list(emails)
    if not all_emails:
        return 0.0
    score = 0.0
    for e in all_emails:
        expected_category = _infer_expected_category(e)
        expected_resolution = _expected_resolution_for_category(expected_category)
        if e.resolution == expected_resolution:
            score += 1.0
        elif e.resolution == "classify":
            # Partial credit for correct triage even when not final execution action.
            score += 0.6
        elif e.resolution in {"reply", "schedule", "ignore"}:
            score += 0.2
    return score / len(all_emails)


def _resolved_ratio(emails: Iterable) -> float:
    all_emails = list(emails)
    if not all_emails:
        return 0.0
    resolved = sum(1 for e in all_emails if e.resolution in {"classify", "reply", "schedule", "ignore"})
    return resolved / len(all_emails)


def _category_resolution_accuracy(emails: Iterable, category: EmailCategory, expected_resolution: str) -> float:
    targets = [e for e in emails if _infer_expected_category(e) == category]
    if not targets:
        return 1.0
    ok = sum(1 for e in targets if e.resolution == expected_resolution)
    return ok / len(targets)


def grade(obs: ExecAssistObservation) -> float:
    all_emails = _collect_all_emails(obs)
    total_emails = len(all_emails)
    if total_emails == 0:
        return 0.0

    processed_ratio = len(obs.processed_emails) / total_emails
    resolved_ratio = _resolved_ratio(all_emails)
    classification_accuracy = compute_classification_score(all_emails)
    action_accuracy = _action_accuracy(all_emails)
    classify_coverage = sum(1 for e in all_emails if e.resolution == "classify") / total_emails
    conflict_count = _count_conflicts(obs)
    conflict_score = max(0.0, 1.0 - min(conflict_count * 0.25, 1.0))
    meeting_acc = _category_resolution_accuracy(all_emails, EmailCategory.MEETING_REQUEST, "schedule")
    query_acc = _category_resolution_accuracy(all_emails, EmailCategory.GENERAL_QUERY, "reply")
    spam_acc = _category_resolution_accuracy(all_emails, EmailCategory.SPAM, "ignore")
    urgent_acc = _category_resolution_accuracy(all_emails, EmailCategory.URGENT_TASK, "classify")

    # Bottleneck term: low performance on any intent category should strongly
    # limit hard-task scores. Geometric mean is stricter than arithmetic mean.
    category_balance = (meeting_acc * query_acc * spam_acc * urgent_acc) ** 0.25

    # Base score emphasizes decision quality and coverage.
    score = (
        0.30 * action_accuracy
        + 0.20 * classification_accuracy
        + 0.05 * classify_coverage
        + 0.20 * resolved_ratio
        + 0.10 * conflict_score
        + 0.15 * category_balance
    )

    # Critical-behavior penalties.
    mis_handled_urgent = sum(
        1
        for e in all_emails
        if _infer_expected_category(e) == EmailCategory.URGENT_TASK and e.resolution == "ignore"
    )
    spam_not_ignored = sum(
        1
        for e in all_emails
        if _infer_expected_category(e) == EmailCategory.SPAM and e.resolution != "ignore"
    )
    meetings_not_scheduled = sum(
        1
        for e in all_emails
        if _infer_expected_category(e) == EmailCategory.MEETING_REQUEST and e.resolution != "schedule"
    )
    queries_not_replied = sum(
        1
        for e in all_emails
        if _infer_expected_category(e) == EmailCategory.GENERAL_QUERY and e.resolution != "reply"
    )

    score -= min(mis_handled_urgent * 0.12, 0.24)
    score -= min(spam_not_ignored * 0.03, 0.09)
    score -= min(meetings_not_scheduled * 0.04, 0.12)
    score -= min(queries_not_replied * 0.03, 0.09)

    unresolved_count = sum(1 for e in all_emails if e.resolution not in {"classify", "reply", "schedule", "ignore"})
    score -= min(unresolved_count * 0.02, 0.16)

    # Distribution penalties: hard task expects mixed workflow behavior,
    # not over-reliance on a single action.
    expected_meetings = sum(1 for e in all_emails if _infer_expected_category(e) == EmailCategory.MEETING_REQUEST)
    expected_replies = sum(1 for e in all_emails if _infer_expected_category(e) == EmailCategory.GENERAL_QUERY)
    expected_spam = sum(1 for e in all_emails if _infer_expected_category(e) == EmailCategory.SPAM)
    expected_urgent = sum(1 for e in all_emails if _infer_expected_category(e) == EmailCategory.URGENT_TASK)

    actual_schedule = sum(1 for e in all_emails if e.resolution == "schedule")
    actual_reply = sum(1 for e in all_emails if e.resolution == "reply")
    actual_ignore = sum(1 for e in all_emails if e.resolution == "ignore")
    actual_classify = sum(1 for e in all_emails if e.resolution == "classify")

    # Penalize deficits relative to expected intent buckets.
    def _deficit(actual: int, expected: int) -> float:
        if expected <= 0:
            return 0.0
        return max(0.0, (expected - actual) / expected)

    score -= 0.05 * _deficit(actual_schedule, expected_meetings)
    score -= 0.04 * _deficit(actual_reply, expected_replies)
    score -= 0.03 * _deficit(actual_ignore, expected_spam)
    score -= 0.03 * _deficit(actual_classify, expected_urgent)

    # Penalize classify-heavy trajectories that avoid concrete workflow actions.
    classify_ratio = actual_classify / total_emails
    if classify_ratio > 0.70:
        score -= min((classify_ratio - 0.70) * 0.20, 0.06)

    # Continuous difficulty scaling by completion and balanced intent handling.
    completion_factor = max(0.0, min(1.0, resolved_ratio))
    score *= (0.85 + 0.15 * completion_factor) * (0.85 + 0.15 * category_balance)

    score = max(0.0, min(1.0, score))
    return round(score, 4)
