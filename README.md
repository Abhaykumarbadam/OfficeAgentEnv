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
tags: [openenv, office, email, scheduling]
---

# OfficeAgentEnv

> **OfficeAgentEnv is an OpenEnv-compliant benchmark for executive-assistant workflows: email triage, response drafting, meeting scheduling, and constraint-aware action selection.**

## Live Demo

- Hugging Face Space: https://huggingface.co/spaces/AbhayBadam09/officeagentenv

## Quick Start

### Local environment

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
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

---

## Overview

OfficeAgentEnv simulates the daily workflow of a corporate executive assistant. The agent receives an inbox of mixed emails and must decide, step by step, how to handle each item.

The environment is designed to test whether an agent can:

- classify incoming messages into meaningful categories,
- draft context-aware replies,
- schedule meetings without conflicts,
- ignore spam without dropping important mail.

This models real tasks performed by millions of workers daily, making it a high-value benchmark for evaluating LLM reasoning, planning, and multi-step decision-making.

## Tasks and Results

| Task | Difficulty | Max Steps | Summary | Current Baseline |
|---|---|---|---|---|
| easy | Easy | 10 | Deterministic classification of 5 emails | ~0.70 |
| medium | Medium | 15 | Classification plus conflict-aware scheduling | ~0.50 |
| hard | Hard | 12 | Full assistant workflow with replies and spam handling | ~0.38 |

These baseline scores are meant as a starting point for comparison. The notebook workflow in this repository is intended to improve them through trajectory collection and supervised fine-tuning.

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
| `hard` | Hard | 12 | Full workflow: classify, reply, schedule, ignore spam | 0.4 |

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
| Per-step cost | -0.02 |

---

## Setup & Usage

### Environment variables

- `HF_TOKEN` (or `OPENAI_API_KEY`): Hugging Face / OpenAI API key (required).
- `API_BASE_URL`: LLM API endpoint, e.g. `https://router.huggingface.co/v1` (required).
- `MODEL_NAME`: Model identifier, e.g. `Qwen/Qwen2.5-72B-Instruct` (required).
- `ENV_URL`: Base URL of the environment HTTP server (optional, defaults to `http://localhost:7860`).

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

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset environment for a task |
| POST | `/step` | Execute one action |
| GET | `/state` | Get current state |
| GET | `/tasks` | List available tasks |
| POST | `/grade` | Score current episode |

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
│   ├── task_hard.py
│   └── scoring.py       # Shared strict (0,1) score utility
└── server/
    └── app.py            # FastAPI server
```

---

## Training Artifacts

The notebook workflow produces the following outputs:

- trained model directory: `/content/officeagent-ft`
- reward curve image: `/content/reward_curve.png`

The notebook is structured so these artifact paths are printed at the end of the run and remain visible in the output.

---

## Reproducibility Notes

- The environment server runs on port `7860`.
- The notebook uses environment-driven configuration where possible.
- The training flow collects trajectories, builds a dataset, fine-tunes the model, and saves artifacts without requiring a separate Python export.
