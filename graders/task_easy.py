"""
task_easy.py — Email Classification Grader
Score = fraction of emails correctly classified (0.0 – 1.0)
"""
from __future__ import annotations
from env.models import EmailCategory, ExecAssistObservation


GROUND_TRUTH = {
    "e001": EmailCategory.MEETING_REQUEST,
    "e002": EmailCategory.SPAM,
    "e003": EmailCategory.URGENT_TASK,
    "e004": EmailCategory.GENERAL_QUERY,
    "e005": EmailCategory.MEETING_REQUEST,
}


def grade(obs: ExecAssistObservation) -> float:
    """
    Checks processed emails against ground-truth labels.
    Returns score in [0.0, 1.0].
    """
    all_emails = obs.processed_emails + obs.pending_emails
    if not GROUND_TRUTH:
        return 0.0

    correct = 0
    total   = len(GROUND_TRUTH)

    for email in all_emails:
        if email.email_id in GROUND_TRUTH:
            if email.category == GROUND_TRUTH[email.email_id]:
                correct += 1

    return round(correct / total, 4)
