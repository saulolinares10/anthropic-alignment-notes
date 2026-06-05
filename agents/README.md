# Study Agents — Anthropic FDE Prep

Two agents to accelerate lesson production. Both share the same curriculum
context and conventions. Use SKILL.md files in Claude Chat, Python agents
in Claude Code.

---

## paper-to-lesson

Converts a paper, article, or notes into a structured lesson draft.

**In Claude Chat:** Drop `paper-to-lesson/SKILL.md` into your project.
Then say: "Use paper-to-lesson on this URL / PDF / text."

**In Claude Code:**
```bash
# From a URL
python agents/paper_to_lesson.py --url https://arxiv.org/abs/2212.08073

# From a PDF
python agents/paper_to_lesson.py --pdf papers/constitutional-ai.pdf

# From pasted text
python agents/paper_to_lesson.py --text "Abstract: We propose..."

# From a notes file
python agents/paper_to_lesson.py --file notes/speculative-decoding-notes.txt
```

Output: `lessons/lesson-NN-topic-name.md`

---

## lesson-builder

Builds a full lesson from a topic, raw notes, or a paper-to-lesson draft.

**In Claude Chat:** Drop `lesson-builder/SKILL.md` into your project.
Then say: "Build lesson 6 on Fine-tuning and PEFT" or
"Turn my notes into lesson 7" or "Expand this draft into a full lesson."

**In Claude Code:**
```bash
# From topic name alone
python agents/lesson_builder.py --lesson 6 --topic "Fine-tuning and PEFT"

# From raw notes
python agents/lesson_builder.py --lesson 7 --topic "Speculative Decoding" \
  --notes notes/speculative-decoding.txt

# Expand a paper-to-lesson draft
python agents/lesson_builder.py --lesson 7 \
  --draft lessons/lesson-draft-speculative-decoding.md

# All three combined (best quality)
python agents/lesson_builder.py --lesson 6 \
  --topic "Fine-tuning and PEFT" \
  --notes notes/peft-notes.txt
```

Output: `lessons/lesson-06-fine-tuning-and-peft.md`

---

## Typical workflow

```
1. Read paper / article
2. paper_to_lesson.py --url <url>       → draft with mechanism + interview angle
3. lesson_builder.py --draft <draft>    → full lesson with project + self-check
4. Review, edit, commit to anthropic-alignment-notes repo
5. Run the hands-on project
6. Answer self-check questions before moving to next lesson
```

---

## Install

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

Model routing (per Cuesta conventions):
- These agents use `claude-sonnet-4-20250514` by default
- For bulk runs (generating many lessons at once), swap to `claude-haiku-4-5-20251001`
- Do not use Opus unless Sonnet fails twice on the same input
