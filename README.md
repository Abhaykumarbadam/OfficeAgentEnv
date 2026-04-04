---
title: OfficeAgentEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
app_port: 7860
pinned: false
---

# OfficeAgentEnv

> **An OpenEnv-compliant real-world environment where an AI agent acts as an executive assistant — processing emails, scheduling meetings, replying to queries, and filtering spam.**

---

## Overview

OfficeAgentEnv simulates the daily workflow of a corporate executive assistant.
The agent receives an inbox of mixed emails and must intelligently decide for each one:
- **Classify** it (meeting request / urgent task / spam / general query)
- **Reply** with a relevant response
- **Schedule** a conflict-free meeting on the calendar
- **Ignore** spam

This models real tasks performed by millions of workers daily, making it a high-value benchmark for evaluating LLM reasoning, planning, and multi-step decision-making.

---

## Action Space

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | enum | ✅ | `classify_email` / `reply_email` / `schedule_meeting` / `ignore_email` |
| `email_id` | string | ✅ | ID of the email to act on |
| `category` | enum | classify only | `meeting_request` / `urgent_task` / `spam` / `general_query` |
| `reply_text` | string | reply only | The reply body text |
| `meeting_title` | string | schedule only | Title of the meeting |
| `meeting_start_time` | string | schedule only | Format: `YYYY-MM-DD HH:MM` |
| `meeting_end_time` | string | schedule only | Format: `YYYY-MM-DD HH:MM` |
| `participants` | list[str] | schedule only | List of participant emails |

---

## Observation Space

```json
{
  "pending_emails":    [...],   // emails not yet processed
  "processed_emails":  [...],   // emails already acted on
  "calendar_events":   [...],   // current scheduled events
  "last_action_result": "...",  // result message from last action
  "current_step":       3,
  "task_name":         "medium"
}
```

---

## Tasks

| Task | Difficulty | Max Steps | Description | Success Threshold |
|---|---|---|---|---|
| `easy` | Easy | 10 | Classify 5 deterministic emails | 0.6 |
| `medium` | Medium | 15 | Classify + schedule meetings from mixed inbox | 0.5 |
| `hard` | Hard | 20 | Full workflow: classify, reply, schedule, ignore spam | 0.4 |

---

## Reward Function

| Action | Reward |
|---|---|
| Correct email classification | +0.30 |
| Conflict-free meeting scheduled | +0.30 |
| Good quality reply (long + relevant) | +0.20 |
| Spam correctly ignored | +0.10 |
| Wrong classification | -0.20 |
| Scheduling conflict | -0.25 |
| Important email ignored | -0.15 |
| Per-step cost | -0.01 |

---

## Setup & Usage

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference
export HF_TOKEN=your_key
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
docker build -t officeagentenv .
docker run -p 7860:7860 officeagentenv
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset environment for a task |
| POST | `/step` | Execute one action |
| GET | `/state` | Get current state |
| GET | `/tasks` | List available tasks |
| POST | `/grade` | Score current episode |

---

## Baseline Scores

| Task | Score | Notes |
|---|---|---|
| easy | ~0.70 | Qwen2.5-72B classifies well |
| medium | ~0.50 | Scheduling requires structured output |
| hard | ~0.38 | Full workflow challenges frontier models |

---

## Project Structure

```
OfficeAgentEnv/
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv spec
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── models.py         # Pydantic types
│   ├── environment.py    # step/reset/state logic
│   ├── email_data.py     # Email data (base + random)
│   └── calendar_data.py
├── graders/
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
└── server/
    └── app.py            # FastAPI server
```
