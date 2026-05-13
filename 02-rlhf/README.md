# 02 — RLHF

**Date:** May 2026
**Paper:** Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022)
**Status:** Complete

## What's Here

| File | Description |
|------|-------------|
| [notebook/rlhf_simulation.ipynb](notebook/rlhf_simulation.ipynb) | Reward model simulation, verbosity bias demo, sycophancy demo, KL penalty intuition |

## Core Idea

RLHF bridges next-token prediction and human preference via three stages: supervised fine-tuning (SFT) on human-written demonstrations establishes a baseline aligned model; a reward model trained on human preference pairs learns to predict which responses humans prefer; PPO fine-tuning then optimizes the policy against the reward model with a KL divergence penalty to prevent reward hacking. The KL penalty is what keeps the model from discovering degenerate solutions that score well on the proxy signal but diverge from actual quality — without it, reward hacking emerges within thousands of steps.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **SFT** | Supervised fine-tuning on human demonstrations — establishes the baseline policy |
| **Reward model** | Trained on preference pairs to predict human-preferred responses as a scalar |
| **PPO** | Proximal Policy Optimization — the RL algorithm used for fine-tuning |
| **KL divergence penalty** | Penalizes the policy for straying too far from the SFT baseline; prevents reward hacking |
| **Reward hacking** | The model finds responses that maximize the proxy reward while diverging from true quality |
| **Goodhart's Law** | When a measure becomes a target, it ceases to be a good measure |
| **Verbosity bias** | Annotators equated length with quality; reward model learned to reward elaboration |
| **Sycophancy** | Model validates user assumptions because validation scores well in the training distribution |
| **Rater inconsistency** | Different annotators apply different standards; reward model averages over disagreement |

## How RLHF and CAI Relate

RLHF provides a grounded human preference signal that principles alone can't fully encode — it captures the implicit, contextual judgments that are hard to articulate as rules. Constitutional AI scales what RLHF can't afford with human annotators: it replaces expensive preference labeling with model self-critique guided by explicit principles. Claude uses both. RLHF shapes the base preference model; CAI refines it by targeting failure modes (sycophancy, harmlessness gaps) that human annotation introduces or can't efficiently address.

## FDE Relevance

- **Verbosity bias → system prompt fix:** Explicit conciseness instructions partially override the verbosity prior, but you're fighting a strong signal. Domain-specific eval harnesses that measure response length vs. task complexity reveal the gap more precisely than intuition.
- **Rater pool mismatch → domain evals:** RLHF was trained on general internet annotation. If your deployment domain has different quality criteria (financial accuracy, medical precision), the reward model's definition of "helpful" may not match your users' needs. Measure directly.
- **Reward hacking → why models sound confident when they shouldn't:** The model learned that confident-sounding responses score well. In ambiguous domains, this produces responses that are fluent and assertive about things the model has low certainty on — a direct artifact of proxy optimization.

## What I'd Build Next

A reward model scoring harness: a fixed evaluation prompt set (20–30 prompts spanning domains), a scoring function that evaluates responses across the four RLHF criteria, and a before/after comparison that measures how much operator context (system prompt changes) shifts the effective reward signal. Hypothesis: a well-crafted system prompt moves composite helpfulness scores by 1–2 points on a 10-point scale. That's measurable, and measuring it gives you a principled way to compare prompt variants rather than relying on vibe checks.
