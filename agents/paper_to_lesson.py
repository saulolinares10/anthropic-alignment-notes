#!/usr/bin/env python3
"""
paper-to-lesson agent
Converts a research paper or article into a structured curriculum lesson note.

Usage:
    python paper_to_lesson.py --url https://arxiv.org/abs/2212.08073
    python paper_to_lesson.py --pdf path/to/paper.pdf
    python paper_to_lesson.py --text "paste abstract or excerpts here"
    python paper_to_lesson.py --file path/to/notes.txt

Output: lesson draft markdown file written to ./lessons/
"""

import anthropic
import argparse
import sys
import re
from datetime import date
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
OUTPUT_DIR = Path("lessons")

CURRICULUM_CONTEXT = """
You are helping Saulo Linares, a PE data lead at Cuesta Partners, prepare for an
Anthropic Forward Deployed Engineer (FDE) role. He runs a structured self-study
curriculum and commits lesson notes to his anthropic-alignment-notes GitHub repo.

Completed lessons:
- Lesson 1: Constitutional AI
- Lesson 2: RLHF (InstructGPT, reward modeling, SFT)
- Lesson 3: Transformers (attention, tokenization, KV cache)
- Lesson 4: RAG & GraphRAG (chunking, CRAG, RAGAS)
- Lesson 5: Multi-agent systems (orchestrator, sub-agents, Colombia election simulator)

His live projects:
- FinMentor: FastAPI + Claude API, 4 financial advisor personas, React Native
- BODEGA AI: WhatsApp inventory agent, multi-turn state, tool use
- M&A Diligence Intelligence System: multi-agent (Claude API), sub-agents for
  financial parsing, IC memo drafting, red-flag detection (customer concentration,
  churn, NWC anomalies, EBITDA adjustments), full observability

Cuesta conventions: Haiku for bulk tasks, Sonnet as default, Opus only if Sonnet
fails twice. XML tags for prompts (role/context/task/format). Schema + anonymized
samples only — never production data.
"""

SYSTEM_PROMPT = f"""
{CURRICULUM_CONTEXT}

When given a paper, article, or set of notes, produce a lesson draft in the
following exact markdown structure. Do not add or remove sections.

# Lesson N — [Topic Name]
> Source: [full citation or URL]
> Date: {date.today().isoformat()}
> Status: draft

---

## Core mechanism

[3–5 paragraphs. Plain English first, technical precision in the final paragraph.
No jargon without immediate gloss.]

---

## Key equations or algorithms

[Only if the paper has meaningful math. For each: one plain sentence on what it
computes, the expression, then term-by-term explanation. If none: write
"No core equations — this is primarily an architectural or systems contribution."]

---

## Connections to prior lessons

[Bullet list. Format: **Lesson X — Topic:** specific connection. Only earned
connections — not generic "also uses transformers".]

---

## Anthropic interview angle

[2–3 paragraphs. What does an interviewer want to hear? What separates someone
who read the abstract from someone who understood the paper? Include one concrete
tradeoff or failure mode the paper surfaces.]

---

## Open questions

[3–5 bullet points. Real unresolved questions: gaps in the method, scaling
unknowns, assumptions that may not hold in production.]

---

## Project anchor

[One paragraph. Specific connection to FinMentor, BODEGA AI, or M&A Diligence
Intelligence System. Not generic — name the exact sub-agent, pipeline stage, or
design decision where this concept applies.]

---

Tone rules:
- Dual-track: plain English first, technical precision second in each section
- Never overclaim. Surface real limitations.
- Target 600–900 words across all sections.
- For the lesson number: infer from the topic if possible, otherwise use N.
"""

# ── Input handlers ─────────────────────────────────────────────────────────────

def load_from_url(url: str, client: anthropic.Anthropic) -> str:
    """Fetch article content via Claude's web fetch (via a fetch prompt)."""
    print(f"Fetching: {url}")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"Please fetch and return the full text content of this URL, "
                       f"preserving all technical content: {url}"
        }],
        tools=[{"type": "web_search_20250305", "name": "web_search"}]
    )
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    return "\n".join(text_blocks)


def load_from_pdf(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a PDF file."""
    import base64
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, "application/pdf"


def load_from_text(text: str) -> str:
    return text


def load_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Output ─────────────────────────────────────────────────────────────────────

def slugify(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s-]", "", title)
    title = re.sub(r"\s+", "-", title.strip())
    return title[:60]


def save_lesson(content: str, topic_hint: str = "") -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = slugify(topic_hint) if topic_hint else f"lesson-{date.today().isoformat()}"
    # Try to extract lesson number from content
    match = re.search(r"# Lesson (\d+)", content)
    prefix = f"lesson-{match.group(1).zfill(2)}-" if match else "lesson-draft-"
    filename = OUTPUT_DIR / f"{prefix}{slug}.md"
    filename.write_text(content, encoding="utf-8")
    return filename


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    messages = []

    if args.url:
        raw = load_from_url(args.url, client)
        messages.append({
            "role": "user",
            "content": f"Convert this article into a lesson note:\n\nURL: {args.url}\n\n"
                       f"Content:\n{raw}"
        })
        hint = args.url.split("/")[-1].replace("-", " ")

    elif args.pdf:
        b64, media_type = load_from_pdf(args.pdf)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": b64}
                },
                {
                    "type": "text",
                    "text": "Convert this paper into a lesson note following the curriculum format."
                }
            ]
        })
        hint = Path(args.pdf).stem.replace("-", " ").replace("_", " ")

    elif args.text:
        raw = load_from_text(args.text)
        messages.append({
            "role": "user",
            "content": f"Convert these notes/text into a lesson note:\n\n{raw}"
        })
        hint = args.text[:40]

    elif args.file:
        raw = load_from_file(args.file)
        messages.append({
            "role": "user",
            "content": f"Convert this content into a lesson note:\n\n{raw}"
        })
        hint = Path(args.file).stem.replace("-", " ").replace("_", " ")

    else:
        print("Error: provide --url, --pdf, --text, or --file")
        sys.exit(1)

    print("Generating lesson draft...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    content = response.content[0].text
    output_path = save_lesson(content, hint)
    print(f"\n✓ Lesson saved to: {output_path}")
    print(f"  {response.usage.input_tokens} input tokens, "
          f"{response.usage.output_tokens} output tokens")
    print("\nPreview:\n" + "-" * 60)
    print(content[:500] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a paper into a lesson note")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="URL of article or arXiv page")
    group.add_argument("--pdf",  help="Path to PDF file")
    group.add_argument("--text", help="Pasted abstract or excerpt (quoted string)")
    group.add_argument("--file", help="Path to text/notes file")
    run(parser.parse_args())
