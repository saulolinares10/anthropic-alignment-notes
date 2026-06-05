---
name: lesson-builder
description: Use this skill when Saulo wants to build a full structured lesson from a topic name, raw notes, or a paper-to-lesson output. Triggers: "build lesson N", "write up the lesson on X", "turn my notes into a lesson", or after paper-to-lesson produces a draft that needs expanding into a full lesson markdown file ready to commit.
---

## Context

Saulo is preparing for an Anthropic FDE role. Each lesson is a markdown file committed to `anthropic-alignment-notes` repo. Lessons follow a dual-track design: accessible explanation for general audiences, technical depth for engineering-level readers. Each lesson ends with a hands-on project deliverable.

Completed lessons for connection context:
- Lesson 1: Constitutional AI (CAI paper, RLHF survey, helpfulness vs. safety tradeoffs)
- Lesson 2: RLHF (InstructGPT, reward modeling, SFT pipeline)
- Lesson 3: Transformers (attention mechanism, tokenization, context windows, KV cache)
- Lesson 4: RAG & GraphRAG (chunking strategies, CRAG, entity graphs, RAGAS eval)
- Lesson 5: Multi-agent systems (orchestrator pattern, sub-agents, tool use, Colombia election simulator)

Upcoming lessons: Fine-tuning/PEFT (6), Speculative Decoding (7), Context Compression (8), Hybrid Retrieval (9), Agent Tool Calling (10), Multi-model routing (11), LLM Evaluation (12), AI Safety deep-dive (13), Interview Prep (14).

Live projects that lessons should connect to:
- **FinMentor**: FastAPI + Claude API, 4 financial advisor personas, React Native, Google Cloud Run
- **BODEGA AI**: WhatsApp-based inventory agent, multi-turn state, tool integration
- **M&A Diligence Intelligence System**: multi-agent orchestration (Claude API + tool use), sub-agents for financial parsing, competitive benchmarking, IC memo drafting, red-flag detection (customer concentration, churn, NWC anomalies, EBITDA adjustments), full observability

Cuesta Partners conventions (always respected):
- Schema and anonymized samples only — never production data
- Model routing: Haiku for bulk/high-volume, Sonnet as default, Opus only when Sonnet fails twice
- Prompt structure: XML tags with role/context/task/format

---

## Input

Saulo will provide one or more of:
- A topic name and lesson number
- Raw notes (messy, uncompressed — preserve signal, don't summarize away detail)
- A `paper-to-lesson` draft to expand
- A project description to anchor the lesson around

---

## Output Format

Produce a single markdown file. Filename convention: `lesson-NN-topic-name.md`

```markdown
# Lesson N — [Topic Name]
> Curriculum: Anthropic FDE Prep
> Date: [today]
> Status: draft | review | final

---

## What this lesson is about

[2–3 sentences. The plain-language version: what problem does this concept solve, and why does it matter for building production AI systems? No jargon in this section.]

---

## Core concepts

### [Concept 1 name]
[Plain-language explanation — 1–2 paragraphs. Use an analogy if one genuinely helps. Avoid forced analogies.]

**Technical depth:**
[Precise version of the same concept. Architecture details, algorithm, equations if relevant. This is the layer for engineering-savvy readers. Keep it honest about complexity — don't oversimplify the math.]

### [Concept 2 name]
[Repeat pattern. 2–4 concepts per lesson is the right range. More than 4 = split into two lessons.]

---

## How it connects to prior lessons

[Short list, same format as paper-to-lesson: Lesson X → specific connection. Only include earned connections.]

---

## The Anthropic angle

[What does Anthropic specifically care about on this topic? What does an interviewer want to hear that separates deep understanding from surface familiarity? Include at least one tradeoff or design decision this concept forces.]

---

## Hands-on project

**Title:** [Short project name]

**What you build:** [1–2 sentences describing the deliverable]

**Connection to M&A Diligence / FinMentor / BODEGA AI:** [Specific, concrete connection — not generic]

**Steps:**
1. [Step 1]
2. [Step 2]
3. [Step 3 — usually: evaluate or instrument what you built]

**Deliverable:** [GitHub artifact: repo, notebook, README, eval results]

---

## Key terms

[Glossary — 4–8 terms. Format: `**term**: definition in one sentence.` Only terms that are genuinely new in this lesson, not repeats from prior lessons.]

---

## Further reading

[2–4 resources. Each with: title, link if available, one sentence on why it's worth reading for this specific lesson — not generic "good resource on X".]

---

## Open questions for next session

[2–3 questions Saulo should be able to answer after completing the hands-on project. These serve as a self-check before moving to the next lesson.]
```

---

## Generation rules

- **Raw notes as input:** Treat raw notes as uncompressed signal. Don't flatten them into bullet points. Preserve sequencing, recurrences, and tensions the notes surface — these often point to the most important concepts.
- **Length:** 900–1400 words across all sections. More than 1400 = split the lesson.
- **Dual-track is mandatory:** Every core concept must have both a plain-language version and a technical depth block. Never collapse them into one register.
- **Project must be real:** The hands-on project must produce an actual artifact (notebook, script, eval results). Not "explore X" or "read about Y."
- **Connections must be earned:** If a prior lesson connection is generic, remove it. Only include it if there's a specific mechanism or result that links them.
- **Cuesta conventions apply:** Any code examples or project steps must respect the data security rule, model routing rule, and XML prompt structure.
