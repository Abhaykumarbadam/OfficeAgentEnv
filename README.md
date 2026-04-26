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
tags: [openenv, office, email, scheduling, rl, enterprise-agents]
---

# OfficeAgentEnv

OfficeAgentEnv is an OpenEnv-compliant, closed-loop environment for training and evaluating LLM agents on enterprise-style workflows.  
The agent does not operate on a static dataset only: it acts through tools, changes world state, receives rewards, and improves over interaction trajectories.

## Why This Project

Most static benchmarks measure isolated language capability. Real enterprise execution is sequential, constrained, and stateful:

- actions have downstream effects,
- penalties can be delayed,
- observations are partial,
- policies must optimize long-term outcomes, not only local correctness.

OfficeAgentEnv is designed to model this setting in a reproducible way.

## Before vs. After: Manual Workflows and RL Agents

| Aspect | Before Application (Old Way) | After Application (RL Agent) |
| --- | --- | --- |
| Decision Making | Manual, based on guesswork | Intelligent, data-driven |
| Task Assignment | Based on habit or intuition | Based on workload & future impact |
| Email Handling | Checked manually | Automatically analyzed |
| Planning Style | Short-term, reactive | Long-term, strategic |
| Workload Distribution | Unbalanced (overload/idle) | Balanced automatically |
| Error Handling | Repeated human mistakes | Learns from mistakes (reward/penalty) |
| Adaptability | Static (no improvement) | Continuously improves over time |
| Problem Handling | Fixes issues after they occur | Prevents issues before they occur |
| Efficiency | Lower, time-consuming | Higher, optimized decisions |
| Outcome | Missed deadlines, unhappy clients | Timely delivery, better satisfaction |

## Why This Is Beyond Traditional RL Benchmarks

Unlike conventional reinforcement learning problems that operate in fixed, well-defined environments (like games or simulations with clear rules and immediate rewards), this application works in a dynamic, enterprise-like setting where conditions constantly change and outcomes are often delayed. Instead of optimizing a single objective, it balances multiple factors such as workload, efficiency, and client satisfaction. The agent also deals with partial information and long-term consequences, making decisions that affect future states rather than only immediate rewards. This makes it much closer to real organizational decision-making than traditional RL setups.

## Closed-Loop Formulation

At each step:

1. the environment emits observation \(o_t\),
2. the policy chooses action \(a_t\),
3. the environment transitions to \(s_{t+1}\),
4. reward \(r_t\) is returned,
5. trajectory \((o_t, a_t, r_t, o_{t+1})\) is logged.

This yields online interaction data:

\[
\mathcal{D} = \{(o_t, a_t, r_t, o_{t+1})\}_{t=1}^{T}
\]

instead of relying only on pre-collected labels.

## Enterprise World State

The environment maintains evolving enterprise state such as:

- project status and deadlines,
- team load,
- client satisfaction.

This state changes as the agent takes actions (for example assigning tasks or updating projects), creating delayed trade-offs.

## Tasks

| Task | Difficulty | Max Steps | Description | Success Threshold |
|---|---|---|---|---|
| `easy` | Easy | 10 | Classification-focused workflow over deterministic emails | 0.6 |
| `medium` | Medium | 15 | Mixed workflow: classification + scheduling + operational decisions | 0.5 |
| `hard` | Hard | 12 | Full workflow with noisy inbox, reply/schedule/ignore/classify trade-offs | 0.4 |

## Action Space

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | enum | Yes | `classify_email`, `reply_email`, `schedule_meeting`, `ignore_email`, `assign_task`, `query_status`, `update_project` |
| `email_id` | string | Yes | Target email identifier |
| `category` | enum | classify only | `meeting_request`, `urgent_task`, `spam`, `general_query` |
| `reply_text` | string | reply only | Response text |
| `meeting_title` | string | schedule only | Meeting title |
| `meeting_start_time` | string | schedule only | `YYYY-MM-DD HH:MM` |
| `meeting_end_time` | string | schedule only | `YYYY-MM-DD HH:MM` |
| `participants` | list[str] | schedule only | Meeting participants |
| `team` | string | assign only | Team identifier (for example `engineering`, `sales`) |
| `project_id` | string | update only | Project id (for example `P1`, `P2`) |
| `project_status` | enum | update only | `on_track`, `delayed`, `blocked`, `completed` |

## Observation Space

```json
{
  "pending_emails": [...],
  "processed_emails": [...],
  "calendar_events": [...],
  "last_action_result": "...",
  "current_step": 2,
  "task_name": "medium",
  "world_state": {
    "projects": [...],
    "team_load": {...},
    "client_satisfaction": 0.75
  }
}
```

## Reward Model (Event-Based)

Reward is modeled mathematically with bounded event terms:

\[
r_t = w_s e_t^{success} + w_q e_t^{quality} + w_e e_t^{efficiency} - w_v e_t^{violation} - w_d e_t^{delayed}
\]

Where:

- \(e_t^{success} \in [0,1]\): objective progress,
- \(e_t^{quality} \in [0,1]\): output quality/relevance,
- \(e_t^{efficiency} \in [0,1]\): step and resource efficiency,
- \(e_t^{violation} \in [0,1]\): invalid/unsafe/conflicting behavior,
- \(e_t^{delayed} \in [0,1]\): delayed negative impact.

Per-step reward is clipped to \([-1, 1]\) for stability.

Episode return:

\[
G_t = \sum_{k=0}^{T-t} \gamma^k \tilde{r}_{t+k}
\]

Optional normalized reporting score:

\[
\text{Score}_{ep}=100 \cdot \frac{G - G_{\min}}{G_{\max} - G_{\min} + \epsilon}
\]

## Policy Training Context (PPO)

The repository includes PPO-based training notebooks for policy improvement over collected trajectories.  
Core objective used in PPO:

\[
L^{CLIP}(\theta)=\mathbb{E}_t \left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
\]

with:

\[
r_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}
\]

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Reset environment for a selected task |
| POST | `/step` | Execute one action |
| GET | `/state` | Get environment state summary |
| GET | `/tasks` | List available tasks |
| POST | `/grade` | Score current episode |

## Quick Start

### Local

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

## Environment Variables

- `HF_TOKEN` or `OPENAI_API_KEY`: LLM provider key.
- `API_BASE_URL`: LLM API endpoint.
- `MODEL_NAME`: model id to use in inference.
- `LOCAL_MODEL_PATH`: optional. If unset, `inference.py` auto-loads `./trained_model` when it contains `config.json`, tokenizer files, and weights (`model.safetensors` or sharded / `pytorch_model.bin`).
- `LOCAL_TOKENIZER_FALLBACK`: optional Hub model id (e.g. for matching `vocab_size`) only if the local `tokenizer.json` cannot be loaded; prefer upgrading `tokenizers` per `requirements.txt` instead.
- `ENV_URL`: environment server URL (default `http://localhost:7860`).

For local checkpoints, use recent `transformers` / `tokenizers` as in `requirements.txt` so `tokenizer.json` (often exported for `tokenizers>=0.20`) loads correctly.

## Project Structure

```text
OfficeAgentEnv/
├── inference.py
├── openenv.yaml
├── README.md
├── MINIBLOG.md
├── env/
│   ├── models.py
│   ├── environment.py
│   ├── email_data.py
│   └── calendar_data.py
├── graders/
│   ├── task_easy.py
│   ├── task_medium.py
│   ├── task_hard.py
│   └── scoring.py
├── server/
│   └── app.py
└── ppo_training_officeagent.ipynb
```

## Notes

- `easy` mode is classification-focused.
- `medium` and `hard` require mixed tool behavior.
- Delayed penalties and partial observability are intentional design choices to train long-horizon policies.

## References

- OpenEnv: https://github.com/huggingface/openenv
- Live demo: https://huggingface.co/spaces/AbhayBadam09/officeagentenv
