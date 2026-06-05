#!/usr/bin/env python3
"""
lesson-builder agent
Builds a full structured lesson markdown file from a topic, raw notes,
or a paper-to-lesson draft.

Usage:
    python lesson_builder.py --topic "Fine-tuning and PEFT" --lesson 6
    python lesson_builder.py --notes path/to/raw_notes.txt --lesson 6
    python lesson_builder.py --draft path/to/paper_to_lesson_draft.md --lesson 7
    python lesson_builder.py --topic "Speculative Decoding" --lesson 7 --notes notes.txt

Output: full lesson markdown written to ./lessons/
"""

import anthropic
import argparse
import re
import sys
from datetime import date
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
OUTPUT_DIR = Path("lessons")

CURRICULUM_STATE = """
COMPLETED LESSONS:
- Lesson 1: Constitutional AI — CAI paper, RLHF survey, helpfulness vs. safety tradeoffs
- Lesson 2: RLHF — InstructGPT, reward modeling, SFT pipeline, preference data
- Lesson 3: Transformers — attention mechanism, BPE tokenization, context windows, KV cache
- Lesson 4: RAG & GraphRAG — chunking strategies, CRAG confidence scoring, entity graphs, RAGAS eval
- Lesson 5: Multi-agent systems — orchestrator pattern, sub-agents, Colombia election simulator capstone

UPCOMING LESSONS (for connection mapping):
6: Fine-tuning & PEFT | 7: Speculative Decoding | 8: Context Compression
9: Hybrid Retrieval | 10: Agent Tool Calling | 11: Multi-model routing
12: LLM Evaluation | 13: AI Safety deep-dive | 14: Interview Prep

LIVE PROJECTS:
- FinMentor: FastAPI + Claude API, 4 financial advisor personas (Bogle, Dalio, Lynch, LATAM),
  React Native mobile frontend, Google Cloud Run
- BODEGA AI: WhatsApp-based inventory agent, multi-turn state, tool integration,
  Latin American SMB focus
- M&A Diligence Intelligence System (in development): multi-agent orchestration (Claude API + tool use),
  specialized sub-agents for: financial parsing, competitive benchmarking, IC memo drafting;
  automated red-flag detection: customer concentration, churn acceleration, NWC anomalies,
  EBITDA adjustments; full observability layer

CUESTA PARTNERS CONVENTIONS:
- Data security: schema and anonymized samples only, never production data
- Model routing: Haiku for bulk/high-volume, Sonnet as default, Opus only when Sonnet fails twice
- Prompt structure: XML tags with role/context/task/format
- Infrastructure: SKILL.md, CLAUDE.md, context.md files across Claude Chat/Code/Cowork
"""

SYSTEM_PROMPT = f"""
You are a curriculum architect helping Saulo Linares build structured lesson notes
for his Anthropic FDE interview preparation. You know his full curriculum state,
live projects, and conventions.

{CURRICULUM_STATE}

Produce a complete lesson markdown file in exactly this structure:

---

# Lesson {{N}} — {{Topic Name}}
> Curriculum: Anthropic FDE Prep
> Date: {date.today().isoformat()}
> Status: draft

---

## What this lesson is about

[2–3 sentences. Plain language. What problem does this concept solve, and why does
it matter for building production AI systems? Zero jargon here.]

---

## Core concepts

### [Concept 1 name]

[Plain-language explanation — 1–2 paragraphs. Analogy only if it genuinely helps.]

**Technical depth:**
[Precise version: architecture details, algorithm, equations if relevant. Honest about
complexity — don't flatten the math.]

### [Concept 2 name]

[Same pattern. 2–4 concepts per lesson. More than 4 = flag that lesson should split.]

---

## How it connects to prior lessons

[Bullet list. **Lesson X — Topic:** specific connection, 1–2 sentences.
Only earned connections.]

---

## The Anthropic angle

[What does an Anthropic interviewer specifically want to hear on this topic?
What separates someone who read the docs from someone who understands the tradeoffs?
Include at least one concrete design decision this concept forces.]

---

## Hands-on project

**Title:** [short name]

**What you build:** [1–2 sentences]

**Connection to M&A Diligence / FinMentor / BODEGA AI:**
[Specific — name the exact sub-agent, pipeline stage, or design decision]

**Steps:**
1. [Step — specific enough to execute]
2. [Step]
3. [Step — usually: evaluate or instrument what you built]

**Deliverable:** [GitHub artifact: notebook, script, eval results, README]

---

## Key terms

[4–8 terms. **term**: one sentence definition. Only new terms for this lesson.]

---

## Further reading

[2–4 resources. Title + link if available + one sentence on why for THIS lesson.]

---

## Self-check questions

[2–3 questions Saulo should answer after completing the project. These are the
"did I actually learn this" gate before moving to the next lesson.]

---

GENERATION RULES:
- Raw notes = uncompressed signal. Preserve sequencing and tensions. Don't flatten.
- Length: 900–1400 words total. Flag if the topic needs a split.
- Dual-track is mandatory: every concept has plain + technical depth.
- Project must produce a real artifact. No "explore" or "read about" tasks.
- Connections must be specific. Remove any generic link.
- Cuesta conventions apply to all code examples and project steps.
- Honest framing: surface real limitations, don't overclaim.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:60]


def save_lesson(content: str, lesson_num: int, topic: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = slugify(topic)
    filename = OUTPUT_DIR / f"lesson-{str(lesson_num).zfill(2)}-{slug}.md"
    filename.write_text(content, encoding="utf-8")
    return filename


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Main ───────────────────────────────────────────────────────────────────────

def build_user_message(args: argparse.Namespace) -> str:
    parts = [f"Build a full lesson for Lesson {args.lesson}."]

    if args.topic:
        parts.append(f"Topic: {args.topic}")

    if args.notes:
        notes = read_file(args.notes)
        parts.append(f"\nRaw notes (preserve signal, don't flatten):\n\n{notes}")

    if args.draft:
        draft = read_file(args.draft)
        parts.append(f"\npaper-to-lesson draft to expand:\n\n{draft}")

    if not args.topic and not args.notes and not args.draft:
        print("Error: provide at least --topic, --notes, or --draft")
        sys.exit(1)

    return "\n".join(parts)


def run(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    user_message = build_user_message(args)

    print(f"Building Lesson {args.lesson}...")
    if args.topic:
        print(f"Topic: {args.topic}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    content = response.content[0].text
    topic_for_slug = args.topic or f"lesson-{args.lesson}"
    output_path = save_lesson(content, args.lesson, topic_for_slug)

    print(f"\n✓ Lesson saved to: {output_path}")
    print(f"  {response.usage.input_tokens} input tokens, "
          f"{response.usage.output_tokens} output tokens")
    print("\nPreview:\n" + "-" * 60)
    print(content[:600] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a full lesson markdown file")
    parser.add_argument("--lesson", type=int, required=True,
                        help="Lesson number (e.g. 6)")
    parser.add_argument("--topic",  help="Topic name (e.g. 'Fine-tuning and PEFT')")
    parser.add_argument("--notes",  help="Path to raw notes file")
    parser.add_argument("--draft",  help="Path to paper-to-lesson draft to expand")
    run(parser.parse_args())
