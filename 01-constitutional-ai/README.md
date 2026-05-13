# 01 — Constitutional AI

Constitutional AI (CAI) is a technique developed at Anthropic to train AI systems that are both helpful and harmless without requiring human feedback on every response. The core mechanism is a two-phase process: **SL-CAI** (supervised learning) uses a critique-revise loop where the model generates an initial response, critiques it against a set of principles (the "constitution"), and rewrites it — producing self-supervised training data. **RLAIF** (reinforcement learning from AI feedback) then trains a reward model using AI-generated preference labels rather than human ones, making the process scalable in a way pure RLHF is not. RLHF is bottlenecked by human annotator throughput and consistency; CAI replaces the most expensive annotation step with model self-critique, enabling alignment at scale.

## What's Here

| File | Description |
|------|-------------|
| [notebook/cai_simulation.ipynb](notebook/cai_simulation.ipynb) | Working SL-CAI loop using the Anthropic Python SDK |
| [POST.md](POST.md) | LinkedIn post draft (~250 words) |

## Key Concepts

| Concept | Description |
|---------|-------------|
| **SL-CAI** | Supervised learning phase: generate → critique → revise loop produces (prompt, revised_completion) training pairs |
| **RLAIF** | RL from AI feedback: the model labels preference pairs, replacing human annotators for the reward model |
| **Constitution** | A list of principles (harmlessness, honesty, safety priority) that guide each critique step |
| **Sycophancy reduction** | CAI improves over RLHF partly by reducing sycophantic outputs — the model critiques against principles, not human approval signals |

## FDE Relevance

System prompts act as a mini-constitution: every principle you encode is a rule the model applies when evaluating its own potential outputs at inference time. Understanding CAI explains *why* well-structured system prompts meaningfully change model behavior — not just surface style, but the optimization target itself. An FDE writing system prompts is, in a real sense, writing a runtime constitution.

## What I'd Build Next

An eval harness that measures the **harmfulness score per revision cycle** — running a judge model on each intermediate response (initial → revised_1 → revised_2 → …) to quantify how much safety signal each cycle adds. Hypothesis: cycle 1 reduces harm score by ~70%; subsequent cycles produce diminishing returns. This would empirically validate the paper's claim about self-critique efficiency and give a principled answer to "how many revision cycles is enough?"
