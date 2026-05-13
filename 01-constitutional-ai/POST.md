# LinkedIn Post Draft

---

I've been building on the Claude API but treating it as a black box — so I'm going deeper.

Last week I rebuilt Anthropic's Constitutional AI (CAI) loop from scratch using the Python SDK, and it changed how I think about system prompts.

**The RLHF bottleneck**
Traditional RLHF requires human annotators to label preference pairs at scale. Anthropic identified this constraint early: you can't align a frontier model on human feedback alone when the model improves faster than annotation throughput.

**The CAI solution**
Constitutional AI works in two phases:

1. **SL-CAI** — Generate a response → ask the model to critique it against a principle → ask the model to rewrite it. The (prompt, final revision) pair becomes supervised training data. No human needed.
2. **RLAIF** — Train a reward model on AI-generated preference labels instead of human ones. Same alignment signal, orders of magnitude more scalable.

**What I built**
A working simulation: 3 red-team prompts designed to elicit manipulation, 5 constitutional principles, 2 revision cycles each. The full generate → critique → revise pipeline in ~100 lines of Python using Claude Sonnet.

**One concrete observation**
Cycle 1 does almost all the work. The first critique-revise pass transforms a borderline response substantially. Cycle 2 makes refinements at the margin. The first principle the model sees has outsized influence on the final output.

**The honest limitation**
A finite constitution has blind spots. Novel harms — attack vectors not represented in the principle list — have no catch mechanism. This is part of why the learned reward model in RLAIF ultimately matters more than any static list of rules.

Repo in comments.

#AnthropicAI #AIAlignment #ConstitutionalAI #MachineLearning #LLM #AIEngineering #Python #ResponsibleAI

---

**Suggested first comment:**

```
Full simulation notebook + write-up:
https://github.com/saulolinares10/anthropic-alignment-notes/tree/main/01-constitutional-ai
```
