# Closed-Loop RL for Enterprise LLM Agents

Modern LLM agent systems are often evaluated in static settings: fixed prompts, fixed datasets, and fixed labels. That setup is useful for measuring narrow capabilities, but it does not capture how enterprise workflows actually behave in production. In real organizations, actions affect the world state, constraints evolve over time, and decision quality must be evaluated over sequences rather than isolated steps.

This project introduces a closed-loop reinforcement learning framework in which an LLM learns to operate inside a simulated enterprise environment. Instead of predicting answers on pre-collected data, the agent acts through tools, receives feedback, and gradually improves its policy through repeated interaction. The result is a training loop that is closer to real organizational decision-making under uncertainty.

## What Makes This Different

What makes this work different from many assistant-style projects is that the world does not stay static while the model reasons: every decision reshapes the next state, so the agent is trained to handle consequences, trade-offs, and uncertainty the way real teams experience them.

## Manual Workflows vs. RL-Trained Agent

The following contrast summarizes the shift from conventional, ad hoc operations to a policy learned through environment interaction:

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

![Closed-Loop RL System Overview](./Overview.png)

## Problem Setting

The core objective is to train an LLM-based agent to manage enterprise operations with long-horizon consequences. The environment models organizational factors such as:

- project progress and deadlines,
- team workload and utilization,
- client satisfaction trajectories,
- operational penalties and delayed failures.

At each timestep, the agent receives a partial observation and selects a tool-based action (for example, assigning tasks, querying status, or updating project plans). Because the environment is partially observable and outcomes can be delayed, the policy must reason beyond immediate rewards.

Formally, this is modeled as a partially observable Markov decision process (POMDP):

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, \gamma)
$$

where:
- $s_t \in \mathcal{S}$ is the latent state,
- $o_t \in \mathcal{O}$ is the observed context,
- $a_t \in \mathcal{A}$ is the chosen action,
- $P$ is the transition dynamics,
- $R$ is the reward function,
- $\gamma$ is the discount factor.

## Environment and Interaction Design

Unlike static evaluation suites, the environment is stateful and non-stationary at episode scale. Every action can alter future feasibility and reward density. For example, a locally optimal assignment may reduce near-term backlog while creating downstream overload that decreases delivery quality and client confidence.

The interaction loop is:

**oₜ → aₜ ∼ πθ(aₜ | oₜ) → (sₜ₊₁, rₜ, oₜ₊₁)**

This mechanism generates training data online from behavior itself:

**𝓓 = {(oₜ, aₜ, rₜ, oₜ₊₁)} for t = 1..T**

rather than relying on external, manually labeled trajectories.

<img src="trajectory.png" alt="Example Trajectory with Delayed Penalty" width="100%" />

## Multi-Objective Reward Modeling

A central challenge is reward design. Enterprise decisions are rarely single-objective, so reward should be computed from interpretable event variables rather than fixed ad-hoc constants.

Define binary or bounded event terms at step t:

- eₜ^success ∈ {0,1}: action advances the active objective,
- eₜ^quality ∈ [0,1]: quality of tool output or plan update,
- eₜ^eff ∈ [0,1]: efficiency term (fewer wasted actions, better utilization),
- eₜ^violation ∈ {0,1}: hard constraint violation (overload, conflict, invalid update),
- eₜ^delay ∈ [0,1]: delayed negative impact signal.

Then use an event-weighted reward:

**rₜ = wₛ·eₜ^success + w_q·eₜ^quality + wₑ·eₜ^eff − wᵥ·eₜ^violation − w_d·eₜ^delay**

where wₛ, w_q, wₑ, wᵥ, w_d ≥ 0 are calibrated coefficients. A fair default is to give penalties a slightly higher weight than any single positive term so unsafe behavior is consistently discouraged.

To stabilize optimization, clip step rewards to a bounded range:

**r̃ₜ = clip(rₜ, -1, 1)**

The episode return is:

**Gₜ = Σ from k=0 to (T−t) of γᵏ · r̃ₜ₊ₖ**

For reporting, map return to a normalized score so runs are comparable:

**Score(ep) = 100 · (Gₜ − G_min) / (G_max − G_min + ε)**

where G_min and G_max are empirical bounds from evaluation scenarios. This creates a fair 0-100 metric tied directly to reward dynamics instead of arbitrary raw totals.

## PPO-Based Policy Optimization

Training uses Proximal Policy Optimization (PPO), selected for stability and practical performance in high-dimensional policy learning. The LLM serves as the policy backbone (for example, Qwen or Mistral variants), and optimization is performed on interaction rollouts.

The PPO clipped objective is:

**L_clip(θ) = Eₜ[min(rₜ(θ)Âₜ, clip(rₜ(θ), 1−ε, 1+ε)Âₜ)]**

with probability ratio:

**rₜ(θ) = πθ(aₜ | oₜ) / πθ_old(aₜ | oₜ)**

and Âₜ denotes estimated advantage. In practice, this constrains overly large policy updates while still allowing consistent improvement.

Advantage estimation can be implemented with generalized advantage estimation (GAE):

**Âₜ = Σ from l=0 to ∞ of (γλ)ˡ · δₜ₊ₗ,  and  δₜ = rₜ + γV(oₜ₊₁) − V(oₜ)**

which improves credit assignment in long-horizon enterprise scenarios.

## Behavioral Evolution During Training

A key result in this setup is qualitative policy evolution:

- **Early phase:** action selection is noisy, reactive, and myopic.
- **Middle phase:** policy begins recognizing tool-action context and immediate constraint structure.
- **Later phase:** agent exhibits strategic sequencing, better workload balancing, and lower delayed penalties.

This shift is visible in reward trajectories, action distribution changes, and reduced conflict rates in behavioral logs. Importantly, improvements are not limited to immediate reward spikes; the policy becomes more consistent in avoiding future failure modes.

### Training Reward Curve

The PPO reward curve summarizes how episode-level return evolves through training. An upward trend with reduced variance indicates that the policy is improving its long-horizon decision quality rather than exploiting short-term heuristics.

![PPO Reward Curve](./PPO_reward_curve.jpeg)

### Loss per Episode

The loss-per-episode plot shows optimization behavior across training iterations. A stabilizing or gradually decreasing profile generally indicates that policy updates are converging toward a more consistent action distribution.

![Loss Episode Graph](./Loss_episode_graph.jpeg)

## Implementation Notes for the Project

To keep the system aligned with reproducible ML workflows, the training pipeline is organized around:

- simulator-driven rollout generation,
- trajectory buffering for PPO updates,
- policy/value optimization across epochs,
- periodic evaluation on held-out scenario seeds,
- logging of reward components and failure cases.

Recommended implementation details to include in the repository and report:

1. explicit state and observation schema,
2. tool-action schema (including invalid-action handling),
3. reward coefficient table and delayed-penalty logic,
4. PPO hyperparameters (clip range, learning rate, batch size, epochs, γ, λ),
5. evaluation protocol (seeds, episode length, metrics, confidence intervals if available).

## Why This Matters

The broader contribution is methodological: moving from static automation to adaptive agent learning. In enterprise settings, reliable behavior depends on handling uncertainty, delayed effects, and operational trade-offs. A closed-loop RL framework provides a principled way to train for exactly those properties.

This project demonstrates that LLM agents can be optimized not only for local correctness, but for long-term decision quality in dynamic environments. That makes it a practical step toward robust, tool-using enterprise AI systems.

## Key Takeaways

- Closed-loop simulation enables self-generated training data without relying only on fixed datasets.
- Partial observability and delayed penalties make long-horizon planning essential.
- Multi-objective rewards better reflect real enterprise optimization targets than single-score objectives.
- PPO provides a stable training mechanism for improving LLM tool-use policies over repeated interaction.
- Behavioral evidence shows transition from reactive actions to strategic decision-making.
