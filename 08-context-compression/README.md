# Lesson 8 — Context Compression
> Curriculum: Anthropic FDE Prep
> Date: 2026-07-02
> Status: draft

---

## What this lesson is about

LLMs have a finite context window — a hard ceiling on how much text they can process in one call. When real-world inputs exceed that ceiling (long documents, multi-turn conversation histories, large tool outputs), something has to give. Context compression is the set of techniques for deciding what to keep, what to summarize, and what to discard so that the most useful information fits inside the window. Done well, it's invisible to the model; done poorly, it silently degrades output quality by dropping the wrong content.

---

## Core concepts

### The context window as a resource budget

Every token in the context window costs money, consumes latency, and competes with every other token for the model's attention. A 200K-token window sounds generous until you have a 40-page financial report, a 30-turn conversation history, and 10 retrieved document chunks — all arriving simultaneously. The real question is never "can the model handle this?" but "what is the cheapest way to give the model what it needs to answer correctly?"

**Technical depth:**
The cost of a full-context forward pass scales roughly as `O(n²)` in attention (mitigated by KV cache but not eliminated at long sequences). Beyond cost, empirical research shows the "lost in the middle" problem: retrieval accuracy degrades for content in the middle of very long contexts, with the model reliably attending to the start and end but losing signal from the center. This means raw concatenation of all available content is often *worse* than smart compression, not just more expensive. The compression decision is therefore both a cost optimization and a quality optimization.

---

### Summarization-based compression

The simplest form of context compression: replace a long section of text with a shorter summary that preserves the information the model needs downstream. This works well for conversation history (where early turns are often low-information) and for long retrieved documents (where only a few paragraphs are relevant to the current query).

**Technical depth:**
The standard pattern is a two-step pipeline: (1) a smaller, cheaper model (Haiku) compresses each chunk; (2) the target model (Sonnet) receives only the compressed context. LLMLingua (Microsoft, 2023) operationalizes this as token-level compression: a small LM scores each token in the prompt by its perplexity contribution, then drops the lowest-scoring tokens while preserving syntactic structure. Compression ratio of 4–20× is achievable with <5% accuracy loss on reasoning tasks. The critical design choice is *what information to preserve* — for financial analysis, numerical values and entity names must be preserved exactly; prose explanation can be aggressively summarized.

---

### KV cache eviction

The KV cache (Lesson 3) stores attention keys and values for all prior tokens so each new generation step doesn't recompute them. But the cache has a finite capacity tied to GPU VRAM. KV cache eviction is the strategy for deciding which cached entries to drop when the cache is full — the hardware-level analog of context compression at the software level.

**Technical depth:**
Three eviction strategies dominate the literature:
- **Recency (StreamingLLM, 2023):** keep the first few "attention sink" tokens and a sliding window of recent tokens; discard the rest. Very fast, no scoring needed. Fails for tasks requiring long-range dependency.
- **Accumulated attention score (H2O, 2023):** evict the token whose cumulative attention weight across all prior steps is lowest — the "heavy hitter" heuristic. Preserves the tokens the model actually attended to. 20% better than recency on long-document QA at the same cache budget.
- **Scissorhands (2023):** evicts based on attention pattern persistence — tokens that appeared in attention peaks in multiple layers are kept. Empirically better than H2O on summarization tasks.

For CPU-hosted models or edge deployments, eviction strategy matters more than architecture — it's often the primary tuning lever.

---

### Selective context retention for agent pipelines

In multi-agent systems (Lesson 5), each sub-agent call accumulates tool outputs, intermediate results, and error logs in the shared context. By the time the orchestrator synthesizes a final response, the context may contain 10× more tokens than the relevant content. Selective retention is the policy for what each agent passes forward — fundamentally a compression decision baked into the orchestration architecture.

**Technical depth:**
The design pattern is a **context manager** layer between sub-agents: each agent's output is routed through a compression step before being appended to the shared thread. Three strategies compose:
1. **Schema projection:** extract only fields the downstream agent needs (e.g., the IC memo sub-agent doesn't need raw parse logs — only the normalized financial metrics).
2. **Importance scoring:** a classifier scores each sentence/chunk for relevance to the original user query; low-scoring chunks are dropped.
3. **Hierarchical summarization:** long sub-agent outputs are recursively summarized at multiple granularities — the orchestrator sees the executive summary; fine-grained details are available on-demand via retrieval.

In the M&A Diligence system, the financial parsing sub-agent returns a structured JSON object (~500 tokens) from a 40-page document (~20K tokens). That extraction step *is* a context compression step, even if it's not framed that way.

---

## How it connects to prior lessons

- **Lesson 3 — Transformers:** KV cache eviction operates directly on the attention cache structure introduced in Lesson 3. Understanding what the KV cache *is* explains why eviction strategy choices (recency vs. attention-weighted) have different tradeoffs for different task types.
- **Lesson 4 — RAG:** Retrieval is a form of compression — instead of loading the entire knowledge base into context, you load only the top-K chunks. But if retrieved chunks are themselves too long, you need a second compression step before final context assembly. Context compression and RAG compose.
- **Lesson 5 — Multi-agent systems:** Every orchestrator-to-sub-agent handoff is an opportunity for context loss. Selective retention is what prevents an agent pipeline from degrading into a context dump where no individual agent has a coherent view of the task.
- **Lesson 7 — Speculative decoding:** Both speculative decoding and context compression attack the same root problem (inference cost) from different angles. Spec decode reduces the number of target model forward passes; context compression reduces the token count per forward pass. In production, you use both.

---

## The Anthropic angle

Claude's 200K token window is a product decision, not just a technical one — it shifts the compression burden from Anthropic to the user. A client who dumps 150K tokens of uncompressed context into every call is paying 150× the input token cost of a client who compresses to 1K tokens of essential content. An FDE's job is to recognize this pattern and re-architect the prompt before cost becomes a deal-breaker.

The interviewer insight: context compression is a *quality* problem as much as a cost problem. "Lost in the middle" means a client with a 40K-token financial document who isn't compressing is likely getting worse answers than a client who summarizes to 4K tokens — even if they're within the window limit. The recommendation to compress is not a cost-saving suggestion; it's a correctness suggestion.

---

## Hands-on project

**Title:** Context Compression Pipeline for M&A Diligence Financial Parser

**What you build:** A notebook that takes a synthetic 40-page financial report (simulated as a long text), applies three compression strategies (truncation, LLMLingua-style token scoring, and schema projection), and measures answer quality and cost for each approach on a set of M&A diligence questions.

**Connection to M&A Diligence Intelligence System:** The financial parsing sub-agent currently receives raw document text. This project builds the compression layer that sits upstream of it — deciding what to send — and quantifies the quality/cost tradeoff for each strategy. The output directly informs whether to add compression as a production step.

**Steps:**
1. Generate a synthetic 40-page financial report (~15K tokens) with seed=42, including revenue tables, footnotes, and narrative MD&A sections
2. Implement Strategy A: truncation to the first N tokens (baseline)
3. Implement Strategy B: sliding-window summarization with Haiku (compress each 2K-token section to 200 tokens)
4. Implement Strategy C: schema projection — extract only the 8 financial metrics the downstream agent needs
5. For each strategy, send the compressed context + 10 M&A diligence questions to Sonnet; record answers, input tokens, and estimated cost
6. Score each answer against a gold-standard key; plot quality vs. cost tradeoff curve

**Deliverable:** `notebook/notebook.ipynb` with compression comparison charts and a 3-sentence deployment recommendation

---

## Key terms

**Context window:** The maximum number of tokens an LLM can process in a single forward pass; both a quality constraint (attention degrades at extremes) and a cost constraint (input tokens are billed).

**Lost in the middle:** Empirical finding that LLMs reliably attend to content at the beginning and end of long contexts but lose signal from the middle — motivating compression over raw concatenation.

**KV cache eviction:** The strategy for dropping cached attention key-value pairs when the cache is full; analogous to context compression at the hardware memory level.

**LLMLingua:** A token-level compression method that scores each token in the prompt by its perplexity contribution and drops low-scoring tokens, achieving 4–20× compression with minimal accuracy loss.

**Selective retention:** The policy in an agent pipeline for what each sub-agent passes forward to the next — the orchestration-level expression of context compression.

**Schema projection:** A hard compression step that discards all content except the fields a downstream agent explicitly needs; highest compression ratio, zero information loss for the named fields, total loss for everything else.

**Hierarchical summarization:** Recursive summarization at multiple granularities — executive summary at the top, detailed content available on-demand — used when both high-level and fine-grained access patterns are needed.

---

## Further reading

- **Liu et al. (2023) — "Lost in the Middle: How Language Models Use Long Contexts"** ([arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)): The empirical paper that named and quantified the lost-in-the-middle problem — read the position bias figures; they explain *why* raw context concatenation degrades quality and make the compression argument concrete.
- **Jiang et al. (2023) — LLMLingua** ([arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736)): The practical compression method most likely to appear in production; read the compression ratio vs. accuracy curves in Section 4 to calibrate what ratio is safe for which task types.
- **Zhang et al. (2023) — H2O: Heavy-Hitter Oracle for KV Cache Eviction** ([arxiv.org/abs/2306.14048](https://arxiv.org/abs/2306.14048)): The attention-weight eviction strategy; read alongside StreamingLLM to understand the recency vs. importance tradeoff in cache management.
- **Anthropic docs — Prompt caching** ([docs.anthropic.com](https://docs.anthropic.com)): Anthropic's prompt caching feature is a form of context reuse that complements compression — read it to understand when caching a compressed context across multiple calls is cheaper than recompressing each time.

---

## Open questions for next session

1. After running the project: which compression strategy produced the best quality/cost tradeoff for M&A diligence questions? Did the answer depend on question type (numerical extraction vs. qualitative assessment)?
2. Schema projection achieves near-zero information loss for the named fields — but who decides which fields to name? What happens when the downstream agent needs a field the schema didn't anticipate? How would you design for this in production?
3. Prompt caching and context compression are both ways to reduce per-call token cost. Under what conditions does caching dominate (i.e., when should you *not* compress but instead cache the full context)?
