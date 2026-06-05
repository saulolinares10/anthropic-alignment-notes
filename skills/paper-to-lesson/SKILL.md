---
name: paper-to-lesson
description: Use this skill when Saulo provides a research paper, article, or technical post to convert into a curriculum lesson. Triggers: any mention of a paper title, arXiv link, dev.to URL, or uploaded PDF in the context of the Anthropic prep curriculum. Input can be pasted abstract/excerpts, a URL to fetch, or an uploaded PDF. Outputs a structured lesson note in Saulo's curriculum format.
---

## Context

Saulo is preparing for an Anthropic Forward Deployed Engineer (FDE) role through a structured self-study curriculum. He has completed Lessons 1–5 (Constitutional AI, RLHF, Transformers, RAG/GraphRAG, Multi-agent systems). Each lesson produces a markdown note committed to his `anthropic-alignment-notes` repo.

His curriculum prioritizes:
- Technical depth with plain-language accessibility (dual-track)
- Explicit connection to Anthropic interview relevance
- Connections to prior lessons and his live projects (FinMentor, BODEGA AI, M&A Diligence Intelligence System)
- Honest framing — no overclaiming, surface real limitations and open questions

---

## Input Handling

The paper can arrive in three forms. Handle each:

**Pasted text / abstract:** Use what's provided. If only an abstract is given, note this explicitly and flag where deeper reading would change the analysis.

**URL:** Fetch the full content. If the URL is a dev.to or blog post, parse the article body. If it's an arXiv abstract page, note that the full PDF would add more depth.

**Uploaded PDF:** Extract key sections: abstract, introduction, method, results/conclusion. Prioritize the method section for technical mechanism extraction.

---

## Output Format

Produce a markdown file structured exactly as follows. Do not add sections not listed here. Do not remove any section.

```markdown
# Lesson N — [Topic Name]
> Source: [full citation or URL]
> Date: [today's date]
> Status: draft

---

## Core mechanism

[3–5 paragraphs. Explain what the paper/post actually does or proposes. Write for a smart non-specialist first — no jargon without immediate plain-English gloss. Then in the final paragraph, add the technical precision layer: architecture choices, key design decisions, why they matter.]

---

## Key equations or algorithms

[Only include if the paper has meaningful math or pseudocode. For each: state what it computes in one plain sentence, then show the expression or algorithm, then explain each term. If no meaningful equations exist, write: "No core equations — this is primarily an architectural or systems contribution."]

---

## Connections to prior lessons

[Explicit callouts to Lessons 1–N already completed. Format as a short list:
- **Lesson X — [Topic]:** [1–2 sentences on how this paper extends, contradicts, or applies that lesson's concepts]
Each connection must be substantive — no generic "this also uses transformers" links.]

---

## Anthropic interview angle

[2–3 paragraphs. Answer: if an Anthropic interviewer asks about this topic, what do they actually want to hear? What's the insight that separates a candidate who read the abstract from one who understood the paper? Include at least one concrete tradeoff or failure mode the paper surfaces.]

---

## Open questions

[3–5 bullet points. Real unresolved questions this paper raises — gaps in the method, scaling unknowns, assumptions that might not hold in production. These are for Saulo's own thinking, not for the interviewer.]

---

## Project anchor

[One paragraph. How does this concept apply to one of Saulo's live projects: FinMentor, BODEGA AI, or M&A Diligence Intelligence System? Be specific — not "you could use this in RAG" but "the context compression pattern maps directly to the IC memo drafting sub-agent in M&A Diligence, where the financial parsing output can exceed 8K tokens before reaching the narrative generation step."]
```

---

## Tone and style rules

- Dual-track: plain English first, technical precision second within each section
- Never overclaim. If the paper's claims are narrow, say so.
- Connections to prior lessons must be earned — only include them if genuinely relevant
- The interview angle should be honest about what's hard, not a polished sales pitch
- Length target: 600–900 words total across all sections
