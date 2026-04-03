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
        "subject": "Can we meet Thursday at 3pm?",
        "body": "Hi, I'd like to schedule a 30-minute sync on Thursday at 3pm to discuss Q3 roadmap. Does that work for you?",
        "timestamp": "2024-07-01 09:00",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "e002",
        "sender": "bob@spam.net",
        "subject": "You've WON a $1000 gift card!!!",
        "body": "Congratulations! Click here to claim your prize. Limited time offer!",
        "timestamp": "2024-07-01 09:05",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "e003",
        "sender": "ceo@corp.com",
        "subject": "URGENT: Server down in production",
        "body": "The main production server is returning 500 errors. We need this fixed within the hour. All hands on deck.",
        "timestamp": "2024-07-01 09:10",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "email_id": "e004",
        "sender": "diana@partner.org",
        "subject": "Question about the API integration",
        "body": "Hello, I was wondering if your API supports OAuth 2.0 for authentication? Could you point me to the documentation?",
        "timestamp": "2024-07-01 09:15",
        "category": EmailCategory.GENERAL_QUERY,
    },
    {
        "email_id": "e005",
        "sender": "hr@corp.com",
        "subject": "Team standup moved to 10am",
        "body": "Just a heads up — tomorrow's standup has been moved from 9am to 10am due to a conflict. Please update your calendars.",
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
        "subject": "Quick call this week?",
        "body": "Hey, are you free for a 15-minute call on Friday afternoon to go over the sales pipeline?",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "sender": "promo@deals.io",
        "subject": "50% off today only!",
        "body": "Flash sale! Don't miss out on our biggest deal of the year. Click now before it's gone.",
        "category": EmailCategory.SPAM,
    },
    {
        "sender": "ops@corp.com",
        "subject": "CRITICAL: Database backup failed",
        "body": "The nightly database backup job failed at 2am. Immediate investigation required before business hours.",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "sender": "support@client.com",
        "subject": "How do I reset my password?",
        "body": "Hi there, I forgot my password and the reset email doesn't seem to be arriving. Can you help?",
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
        "subject": "Can we find some time tomorrow afternoon?",
        "body": (
            "Hi, I'd like to sync on the Q3 hiring plan sometime tomorrow "
            "afternoon, maybe between 2pm and 5pm. Whatever works best for you."
        ),
        "timestamp": "2024-07-01 09:30",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "h002",
        "sender": "events@marketinghub.io",
        "subject": "Exclusive leadership webinar this week",
        "body": (
            "Join our free leadership webinar with industry experts. "
            "Reserve your spot now and unlock special software discounts."
        ),
        "timestamp": "2024-07-01 09:35",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "h003",
        "sender": "frank@corp.com",
        "subject": "Follow-up on customer escalation + quick sync",
        "body": (
            "Customer ACME escalated a P1 incident yesterday. "
            "Can you summarize the status for me and also grab 30 minutes "
            "later today so we can walk through next steps?"
        ),
        "timestamp": "2024-07-01 09:40",
        "category": EmailCategory.URGENT_TASK,
    },
    {
        "email_id": "h004",
        "sender": "sarah@client.com",
        "subject": "Quarterly review meeting + billing question",
        "body": (
            "I'd like to schedule our quarterly account review sometime next "
            "week and I also have a question about an unexpected line item on "
            "the last invoice. When could we meet, and who should I contact "
            "about billing?"
        ),
        "timestamp": "2024-07-01 09:45",
        "category": EmailCategory.MEETING_REQUEST,
    },
    {
        "email_id": "h005",
        "sender": "newsletter@randomoffers.biz",
        "subject": "Important update to your account",
        "body": (
            "We noticed unusual activity on your account. Click this link "
            "to secure your profile and claim a complimentary security scan."
        ),
        "timestamp": "2024-07-01 09:50",
        "category": EmailCategory.SPAM,
    },
    {
        "email_id": "h006",
        "sender": "support@prospect.com",
        "subject": "Two quick questions about pricing and a demo",
        "body": (
            "Hi, we're evaluating your product. Could you clarify whether the "
            "enterprise plan includes SSO and also let me know if we can get "
            "a 45-minute demo sometime early next week?"
        ),
        "timestamp": "2024-07-01 09:55",
        "category": EmailCategory.GENERAL_QUERY,
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
        # - additional complex hard emails (6)
        # - random augmentation (5) for variety, seeded for determinism
        hard_extras = [Email(**e) for e in HARD_EXTRA_EMAILS]
        random_emails = build_random_emails(n=5, seed=seed)
        return base + hard_extras + random_emails
