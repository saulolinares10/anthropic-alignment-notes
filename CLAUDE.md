# CLAUDE.md — anthropic-alignment-notes

## Skills available
- skills/paper-to-lesson/SKILL.md — converts papers/articles to lesson drafts
- skills/lesson-builder/SKILL.md — builds full lesson markdown from topic/notes/draft

## When to use them
- Any paper, URL, or PDF mentioned → use paper-to-lesson
- "build lesson N" or "write up the lesson" → use lesson-builder

## Conventions
- Model routing: Haiku for bulk, Sonnet default, Opus only if Sonnet fails twice
- Output lessons to: lessons/
- Never use production data — schema and anonymized samples only
