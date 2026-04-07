"""
email_data.py
Provides a deterministic base set of emails + a random augmentation pool.
reset() in environment.py picks from both pools based on task difficulty.
"""
from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from typing import List

from env.models import CalendarEvent, Email, EmailCategory


# ---------------------------------------------------------------------------
# Hardcoded base emails  (always present, deterministic)
# ---------------------------------------------------------------------------

BASE_EMAILS: List[dict] = [
    {
        "email_id": "e001",
        "sender": "alice@corp.com",
        "subject": "Request: 30-minute roadmap alignment this Thursday",
        "body": "Hi team, could we schedule a 30-minute alignment on Thursday at 3:00 PM to review Q3 roadmap dependencies and owners? Please confirm availability.",
        "timestamp": "2024-07-01 09:00",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "e002",
        "sender": "bob@spam.net",
        "subject": "Congratulations! Claim your $1000 gift card now",
        "body": "You have been selected for an exclusive reward. Click the secure link to claim your gift card before the offer expires.",
        "timestamp": "2024-07-01 09:05",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "e003",
        "sender": "ceo@corp.com",
        "subject": "URGENT: Production API outage impacting customers",
        "body": "The production API is returning HTTP 500 errors for multiple enterprise accounts. Please initiate incident response immediately and share an ETA within 30 minutes.",
        "timestamp": "2024-07-01 09:10",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "email_id": "e004",
        "sender": "diana@partner.org",
        "subject": "Question: OAuth 2.0 support in partner API",
        "body": "Hello, can you confirm whether your partner API supports OAuth 2.0 client credentials flow and share the relevant integration documentation?",
        "timestamp": "2024-07-01 09:15",
        "category": EmailCategory.GENERAL_QUERY,
    },
    {
        "email_id": "e005",
        "sender": "hr@corp.com",
        "subject": "Schedule update: Engineering standup moved to 10:00 AM",
        "body": "Please note tomorrow's engineering standup is moved from 9:00 AM to 10:00 AM due to a cross-functional planning conflict. Kindly update your calendars.",
        "timestamp": "2024-07-01 09:20",
        "category": EmailCategory.MEETING_REQUEST,
    },
]

# ---------------------------------------------------------------------------
# Random augmentation pool
# ---------------------------------------------------------------------------

_RANDOM_POOL: List[dict] = [
    {
        "sender": "mark@sales.com",
        "subject": "Request: 15-minute sales pipeline review",
        "body": "Could we schedule a 15-minute call on Friday afternoon to review regional pipeline status and next-quarter forecasts?",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "sender": "promo@deals.io",
        "subject": "Limited-time software discount offer",
        "body": "Exclusive flash discount available today only. Activate your offer through this link before midnight.",
        "category": EmailCategory.SPAM,
    },
    {
        "sender": "ops@corp.com",
        "subject": "CRITICAL: Nightly database backup failure",
        "body": "The 02:00 backup job failed for production databases. Please investigate immediately and provide remediation status before business hours.",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "sender": "support@client.com",
        "subject": "Assistance needed: password reset email not received",
        "body": "I attempted a password reset twice but did not receive any reset email. Could you help troubleshoot this issue?",
        "category": EmailCategory.GENERAL_QUERY,
    },
    {
        "sender": "jenny@design.com",
        "subject": "Design review session",
        "body": "Can we block 1 hour on Wednesday morning to review the new UI mockups with the product team?",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "sender": "lottery@fake.com",
        "subject": "Your inheritance awaits",
        "body": "Dear sir/madam, a distant relative has left you $4.5M. Send us your bank details to claim.",
        "category": EmailCategory.SPAM,
    },
    {
        "sender": "finance@corp.com",
        "subject": "URGENT: Invoice overdue",
        "body": "Vendor invoice #INV-8821 is 30 days overdue. Please approve payment immediately to avoid service suspension.",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "sender": "newuser@gmail.com",
        "subject": "What are your office hours?",
        "body": "Hi, I'm a potential customer and I'd like to know your business hours and location.",
        "category": EmailCategory.GENERAL_QUERY,
    },
]


# ---------------------------------------------------------------------------
# Additional hard-task emails (always included for "hard")
# ---------------------------------------------------------------------------

HARD_EXTRA_EMAILS: List[dict] = [
    {
        "email_id": "h001",
        "sender": "carol@corp.com",
        "subject": "Request: hiring plan review tomorrow afternoon",
        "body": (
            "Hi, I would like to schedule a discussion on the Q3 hiring plan "
            "tomorrow between 2:00 PM and 5:00 PM. A 30-minute slot works."
        ),
        "timestamp": "2024-07-01 09:30",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "h002",
        "sender": "events@marketinghub.io",
        "subject": "Exclusive leadership webinar invitation",
        "body": (
            "Join our leadership webinar and unlock premium software discounts. "
            "Reserve your seat today using the promotional registration link."
        ),
        "timestamp": "2024-07-01 09:35",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "h003",
        "sender": "frank@corp.com",
        "subject": "P1 escalation follow-up and incident sync",
        "body": (
            "Customer ACME has escalated a P1 incident. Please provide a status "
            "summary and coordinate a 30-minute sync today to finalize mitigation steps."
        ),
        "timestamp": "2024-07-01 09:40",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "email_id": "h004",
        "sender": "sarah@client.com",
        "subject": "Quarterly business review scheduling and billing clarification",
        "body": (
            "We would like to schedule a quarterly business review next week. "
            "Additionally, there is an unexpected line item on our latest invoice. "
            "Please advise available meeting windows and billing point of contact."
        ),
        "timestamp": "2024-07-01 09:45",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "h005",
        "sender": "newsletter@randomoffers.biz",
        "subject": "Security notice: verify your account access",
        "body": (
            "We detected unusual account activity. Verify credentials immediately "
            "through this external link and claim a complimentary security scan."
        ),
        "timestamp": "2024-07-01 09:50",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "h006",
        "sender": "support@prospect.com",
        "subject": "Enterprise pricing question and demo request",
        "body": (
            "We are evaluating your platform. Does the enterprise tier include "
            "SSO and audit logs, and can we schedule a 45-minute demo early next week?"
        ),
        "timestamp": "2024-07-01 09:55",
        "category": EmailCategory.GENERAL_QUERY,
    },
    {
        "email_id": "h007",
        "sender": "legal@partnerco.com",
        "subject": "Request: contract review call before signature",
        "body": (
            "Could we schedule a 30-minute call tomorrow to review final MSA redlines "
            "before legal sign-off? Please share available windows."
        ),
        "timestamp": "2024-07-01 10:00",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "h008",
        "sender": "security@corp.com",
        "subject": "CRITICAL: Suspicious login activity detected",
        "body": (
            "Multiple suspicious login attempts were detected on privileged accounts. "
            "Please initiate containment and confirm response actions immediately."
        ),
        "timestamp": "2024-07-01 10:05",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "email_id": "h009",
        "sender": "procurement@vendorhub.net",
        "subject": "Invoice discrepancy clarification needed",
        "body": (
            "We found a discrepancy in PO-7742 billing totals. Could you confirm the "
            "approved amount and the expected payment timeline?"
        ),
        "timestamp": "2024-07-01 10:10",
        "category": EmailCategory.GENERAL_QUERY,
    },
    {
        "email_id": "h010",
        "sender": "offers@globaldeals-mail.com",
        "subject": "Action required: unlock premium account benefits",
        "body": (
            "Your organization is eligible for premium account benefits. Verify your "
            "details using this external form to activate rewards."
        ),
        "timestamp": "2024-07-01 10:15",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "h011",
        "sender": "pm@corp.com",
        "subject": "Planning session for launch readiness",
        "body": (
            "Can we book a 45-minute planning session this week to finalize launch "
            "readiness checklist owners and deadlines?"
        ),
        "timestamp": "2024-07-01 10:20",
        "category": EmailCategory.MEETING_REQUEST,
    },
]

# ---------------------------------------------------------------------------
# Base calendar events (always present)
# ---------------------------------------------------------------------------

BASE_CALENDAR: List[dict] = [
    {
        "event_id": "c001",
        "title": "Weekly Sync",
        "start_time": "2024-07-01 10:00",
        "end_time": "2024-07-01 11:00",
        "participants": ["alice@corp.com", "bob@corp.com"],
    },
    {
        "event_id": "c002",
        "title": "Lunch Break",
        "start_time": "2024-07-01 12:00",
        "end_time": "2024-07-01 13:00",
        "participants": [],
    },
]


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_base_emails() -> List[Email]:
    return [Email(**e) for e in BASE_EMAILS]


def build_random_emails(n: int = 3, seed: int | None = None) -> List[Email]:
    rng = random.Random(seed)
    chosen = rng.sample(_RANDOM_POOL, min(n, len(_RANDOM_POOL)))
    emails = []
    base_dt = datetime(2024, 7, 1, 9, 30)
    for i, e in enumerate(chosen):
        ts = (base_dt + timedelta(minutes=10 * i)).strftime("%Y-%m-%d %H:%M")
        emails.append(Email(
            email_id=f"r{str(uuid.uuid4())[:6]}",
            sender=e["sender"],
            subject=e["subject"],
            body=e["body"],
            timestamp=ts,
            category=e["category"],
        ))
    return emails


def build_calendar_events() -> List[CalendarEvent]:
    return [CalendarEvent(**c) for c in BASE_CALENDAR]


def get_emails_for_task(task_name: str, seed: int | None = None) -> List[Email]:
    """
    easy  → base emails only (5, fully deterministic)
    medium→ base + 2 random
    hard  → base + 5 random (noisy, conflicting)
    """
    base = build_base_emails()
    if task_name == "easy":
        # 5 fully deterministic base emails
        return base
    elif task_name == "medium":
        # base + 2 deterministic random emails
        return base + build_random_emails(n=2, seed=seed)
    else:  # hard
        # Hard task: rich, mixed inbox.
        # - base emails (5)
        # - deterministic sample of complex hard emails (4)
        # - random augmentation (2) for variety, seeded for determinism
        rng = random.Random(seed)
        sampled_hard = rng.sample(HARD_EXTRA_EMAILS, k=min(4, len(HARD_EXTRA_EMAILS)))
        hard_extras = [Email(**e) for e in sampled_hard]
        random_emails = build_random_emails(n=2, seed=seed)
        return base + hard_extras + random_emails
