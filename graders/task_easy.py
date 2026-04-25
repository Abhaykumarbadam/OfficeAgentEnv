"""
task_easy.py — Email Classification Grader
Score = fraction of emails correctly classified (0.0 – 1.0)
"""
from __future__ import annotations
from env.models import EmailCategory, ExecAssistObservation

from graders.scoring import strict_unit_interval


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
    Returns score strictly within (0, 1).
    """
    if not GROUND_TRUTH:
        return strict_unit_interval(0.0)

    correct = 0
    total   = len(GROUND_TRUTH)

    for email in obs.processed_emails:
        if email.email_id in GROUND_TRUTH:
            if email.resolution == "classify" and email.category == GROUND_TRUTH[email.email_id]:
                correct += 1

    return strict_unit_interval(correct / total)
